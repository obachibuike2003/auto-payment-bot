# telegram_runner.py
import os, time, json, logging
from threading import Thread
import asyncio


from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

import requests

# Import ONLY the service objects from backend so the service can notify via Telegram
from app import p2p_bot_service, redis_client

logger = logging.getLogger("telegram_runner")
logging.basicConfig(level=logging.INFO)

# Fixes your earlier error: define BACKEND_URL here
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
SUCCESS_RATE_POLL_MIN = int(os.getenv("TELEGRAM_SUCCESS_RATE_EVERY_MIN", "0"))  # 0 = disable auto loop


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")  # string

class TelegramBot:
    def __init__(self, token: str, chat_id: str, p2p_bot_service, redis_client):
        self.awaiting_fix = {}
        self.token = token
        self.chat_id = str(chat_id) if chat_id else None
        self.p2p = p2p_bot_service
        self.redis = redis_client
        self.app = ApplicationBuilder().token(self.token).post_init(self._post_init).build()
        self.list_limit = 10

    async def _safe_send(self, text: str):
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def notify(self, text: str):
        await self._safe_send(text)

    async def send_success_notification(self, order_id, transfer_details: dict):
        text = (
            f"✅ Paid & Confirmed\n"
            f"• Order: {order_id}\n"
            f"• NGN: {transfer_details.get('amount_naira','N/A')}\n"
            f"• Bank: {transfer_details.get('recipient_bank','N/A')}\n"
            f"• Account: {transfer_details.get('recipient_account','N/A')}"
        )
        await self._safe_send(text)

    async def _check_chat(self, update: Update) -> bool:
        if not self.chat_id:
            return True
        return str(update.effective_chat.id) == self.chat_id

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update): return
        msg = (
            "👋 Bot control.\n"
            "/startbot – start scheduler\n"
            "/stopbot – stop scheduler\n"
            "/status – show status\n"
            "/history [n] – last n transfers (default 10)\n"
            "/approve <order_id>\n"
            "/unstuck <order_id>\n"
            "/setapproval on|off\n"
            "/counts – show counters\n"
            "/queue – list pending orders"
        )
        await update.message.reply_text(msg)

    # === control commands that call the backend ===
    async def cmd_startbot(self, update, ctx):
        if not await self._check_chat(update):
            return
        try:
            r = requests.post(f"{BACKEND_URL}/control/start", timeout=10)
            if r.ok:
                data = r.json()
                await update.message.reply_text(
                    f"✅ Started | running={data.get('running')} | job={data.get('job_exists')}"
                )
            else:
                await update.message.reply_text(f"❌ Start failed: {r.status_code} {r.text}")
        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {e}")

    async def cmd_stopbot(self, update, ctx):
        if not await self._check_chat(update):
            return
        try:
            r = requests.post(f"{BACKEND_URL}/control/stop", timeout=10)
            await update.message.reply_text(
                "⏸ Stopped" if r.ok else f"❌ Stop failed: {r.status_code} {r.text}"
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {e}")

    async def cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update):
            return
        try:
            r = requests.get(f"{BACKEND_URL}/control/status", timeout=10)
            if not r.ok:
                await update.message.reply_text(f"❌ Status failed: {r.status_code} {r.text}")
                return

            s = r.json()
            last = s.get("last_cycle")
            last_txt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)) if last else "never"
            txt = (
                f"🧠 Status: {'Running' if s.get('running') else 'Stopped'}\n"
                f"🗒 Job exists: {s.get('job_exists')}\n"
                f"⏱ Scheduler active: {s.get('running')}\n"
                f"⏳ Last cycle: {last_txt}\n"
                f"🛡 Approval mode: {s.get('use_approval_mode', 'N/A')}\n"
                f"🏦 Nomba enabled: {s.get('use_nomba_for_transfers', 'N/A')}"
            )
            await update.message.reply_text(txt)
        except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {e}")

    # === info & control helpers ===
    async def cmd_history(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update): return
        n = 10
        if ctx.args and ctx.args[0].isdigit():
            n = max(1, min(50, int(ctx.args[0])))
        items = self.redis.lrange("p2p_bot:transfers", 0, n-1) if self.redis else []
        if not items:
            await update.message.reply_text("No recent transfers.")
            return
        lines = []
        for raw in items:
            try:
                t = json.loads(raw)
                lines.append(f"• {t.get('bybit_order_id')} – NGN {t.get('amount_naira')} – {t.get('recipient_bank')} {t.get('recipient_account')}")
            except Exception:
                lines.append(f"• {raw[:100]}")
        await update.message.reply_text("🧾 Recent transfers:\n" + "\n".join(lines))

    async def cmd_approve(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update): return
        if not ctx.args:
            await update.message.reply_text("Usage: /approve <order_id>")
            return
        order_id = ctx.args[0]
        res = self.p2p.approve_order(order_id)
        await update.message.reply_text(res.get("message","done"))

    async def cmd_unstuck(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update): return
        if not ctx.args:
            await update.message.reply_text("Usage: /unstuck <order_id>")
            return
        order_id = ctx.args[0]
        res = self.p2p.unstuck_order(order_id)
        await update.message.reply_text(res.get("message","done"))

    async def cmd_setapproval(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update): return
        if not ctx.args or ctx.args[0].lower() not in ("on","off"):
            await update.message.reply_text("Usage: /setapproval on|off")
            return
        on = ctx.args[0].lower() == "on"
        self.p2p.set_approval_mode(on)
        await update.message.reply_text(f"Approval mode set to {on}")

    async def cmd_counts(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update): return
        r = self.redis
        def sc(s): return r.scard(s) if r else 0
        txt = (
            f"📊 Counts\n"
            f"• processed: {sc('p2p_bot:processed_orders')}\n"
            f"• stuck: {sc('p2p_bot:stuck_orders')}\n"
            f"• insufficient: {sc('p2p_bot:insufficient_funds_orders')}\n"
            f"• pending (transfers): {sc('p2p_bot:pending_transfers')}"
        )
        await update.message.reply_text(txt)

    async def send_stuck_order_notification(self, order_id, reason, order_details: dict):
        text = (
            f"❌ Payment failed / stuck\n"
            f"• Order: {order_id}\n"
            f"• Reason: {reason}\n\n"
            "Edit the details below and send them back in this format:\n"
            "Bank: {bank}\n"
            "Account: {account}\n"
            "Name: {name}\n"
        ).format(
            bank=order_details.get('seller_bank_name', 'N/A'),
            account=order_details.get('seller_account_no', 'N/A'),
            name=order_details.get('seller_real_name', 'N/A')
        )
        self.awaiting_fix[order_id] = {"chat_id": self.chat_id, "details": order_details}
        await self._safe_send(text)

    async def handle_fix_reply(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update):
            return
        if not update.message or not update.message.text:
            return

        msg = update.message.text.strip()
        order_id = None
        for oid, info in list(self.awaiting_fix.items()):
            if str(update.effective_chat.id) == str(info["chat_id"]):
                order_id = oid
                break
        if not order_id:
            return

        try:
            lines = [l.strip() for l in msg.splitlines() if l.strip()]
            bank_line = next((l for l in lines if l.lower().startswith("bank:")), None)
            acct_line = next((l for l in lines if l.lower().startswith("account:")), None)
            name_line = next((l for l in lines if l.lower().startswith("name:")), None)
            if not (bank_line and acct_line and name_line):
                raise ValueError("Please send 3 lines starting with Bank:, Account:, Name:")

            new_bank = bank_line.split(":", 1)[1].strip()
            new_account = acct_line.split(":", 1)[1].strip()
            new_name = name_line.split(":", 1)[1].strip()

            if self.redis:
                self.redis.hset(f"p2p_bot:manual_overrides:{order_id}", mapping={
                    "bank": new_bank, "account": new_account, "name": new_name, "ts": str(time.time())
                })

            await update.message.reply_text(
                f"🔄 Retrying order {order_id} with:\n"
                f"Bank: {new_bank}\n"
                f"Account: {new_account}\n"
                f"Name: {new_name}"
            )
            self.p2p.retry_failed_order(order_id, new_bank, new_account, new_name)
            self.awaiting_fix.pop(order_id, None)

        except Exception as e:
            await update.message.reply_text(
                "❌ Couldn’t parse. Send exactly:\n"
                "Bank: ...\nAccount: ...\nName: ...\n\n"
                f"Error: {e}"
            )


    async def cmd_successrate(self, update, ctx):
         if not await self._check_chat(update):
            return
         try:
            r = requests.get(f"{BACKEND_URL}/control/success-rate", timeout=10)
            if not r.ok:
                await update.message.reply_text(f"❌ Failed: {r.status_code} {r.text}")
                return
            data = r.json()
            rate = data.get("success_rate")
            succ = data.get("successful")
            total = data.get("total_attempted")
            if rate is None:
                await update.message.reply_text("No data yet.")
                return
            msg = (
                "📈 *Success Rate*\n"
                f"• Rate: *{rate:.2f}%*\n"
                f"• Successful: *{succ}*\n"
                f"• Total Attempted: *{total}*"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
         except Exception as e:
            await update.message.reply_text(f"⚠️ Error: {e}")

    async def _success_rate_loop(self):
        """
        Optional periodic pings to the chat. Enabled when TELEGRAM_SUCCESS_RATE_EVERY_MIN > 0
        """
        if SUCCESS_RATE_POLL_MIN <= 0:
            return
        while True:
            try:
                r = requests.get(f"{BACKEND_URL}/control/success-rate", timeout=10)
                if r.ok:
                    data = r.json()
                    rate = data.get("success_rate")
                    succ = data.get("successful")
                    total = data.get("total_attempted")
                    if rate is not None:
                        msg = (
                            "⏰ *Auto Report*\n"
                            f"• Success Rate: *{rate:.2f}%*\n"
                            f"• Successful: *{succ}*\n"
                            f"• Total Attempted: *{total}*"
                        )
                        await self._safe_send(msg)
            except Exception as e:
                # log, but don't spam chat
                pass
            await asyncio.sleep(SUCCESS_RATE_POLL_MIN * 60)

    def start_in_thread(self):
        # ... your existing code ...
        self.app.add_handler(CommandHandler("successrate", self.cmd_successrate))
        # ... handlers for others ...

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # kick off the optional auto loop
            loop.create_task(self._success_rate_loop())
            loop.run_until_complete(
                self.app.run_polling(drop_pending_updates=True, stop_signals=None, close_loop=False)
            )

        self.thread = Thread(target=_run, daemon=True)
        self.thread.start()

    # queue + callbacks
    async def cmd_queue(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._check_chat(update):
            return
        await update.message.reply_text("Fetching pending orders…")
        try:
            orders = self.p2p.bybit_api.get_pending_orders()
        except Exception as e:
            await update.message.reply_text(f"Failed to fetch: {e}")
            return
        if not orders:
            await update.message.reply_text("No pending orders found.")
            return

        for o in orders[: self.list_limit]:
            order_id = o.get('orderId') or o.get('id')
            amount   = o.get('fiatAmount') or o.get('amount')
            seller   = o.get('sellerInfo', {}) or {}
            nm = seller.get('accountHolderName', 'N/A')
            bank = seller.get('bankName', 'N/A')
            acc  = seller.get('bankAccountNo', 'N/A')

            text = (
                f"🟡 Pending Order\n"
                f"• ID: {order_id}\n"
                f"• Amount: NGN {amount}\n"
                f"• Name: {nm}\n"
                f"• Bank: {bank}\n"
                f"• Account: {acc}\n"
            )
            kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"approve:{order_id}"),
                    InlineKeyboardButton("🚫 Skip",    callback_data=f"skip:{order_id}"),
                ],
                [ InlineKeyboardButton("📄 Details", callback_data=f"details:{order_id}") ]
            ])
            await update.message.reply_text(text, reply_markup=kb, parse_mode="Markdown")

    async def on_cb_approve(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()
        order_id = q.data.split(":", 1)[1]
        res = self.p2p.approve_order(order_id)
        await q.edit_message_text(q.message.text + "\n\n✅ Approved.", parse_mode="Markdown")

    async def on_cb_skip(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer("Skipped")
        await q.edit_message_text(q.message.text + "\n\n⏭ Skipped.", parse_mode="Markdown")

    async def on_cb_details(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer("Loading…")
        order_id = q.data.split(":", 1)[1]
        try:
            info = self.p2p.bybit_api.get_order_details(order_id)
        except Exception as e:
            await q.edit_message_text(q.message.text + f"\n\n❌ Details error: {e}")
            return

        if not info or info.get("ret_code") != 0:
            await q.edit_message_text(q.message.text + f"\n\n❌ No details: {info and info.get('ret_msg')}")
            return

        res = info.get("result", {})
        terms = res.get("paymentTermList", [])
        lines = [f"- Type {t.get('paymentType')} | {t.get('bankName','?')} {t.get('accountNo','?')}" for t in terms]
        extra = (
            f"\n📄 Details\n"
            f"• Seller: {res.get('sellerRealName','N/A')}\n"
            f"• Terms:\n" + ("\n".join(lines) if lines else "  (none)")
        )
        await q.edit_message_text(q.message.text + extra, parse_mode="Markdown")

    async def _post_init(self, app):
         await app.bot.set_my_commands([
            ("start", "Show bot controls"),
            ("startbot", "Start the scheduler"),
            ("stopbot", "Stop the scheduler"),
            ("status", "Show current status"),
            ("history", "Show recent transfers (/history 10)"),
            ("approve", "Approve an order (/approve <order_id>)"),
            ("unstuck", "Unstuck an order (/unstuck <order_id>)"),
            ("setapproval", "Toggle approval mode (/setapproval on|off)"),
            ("counts", "Show counters"),
            ("queue", "List pending orders"),
            ("successrate", "Show current success rate"),   # <-- add this
        ])

        # Kick off periodic success-rate messages if enabled (no-op when env is 0)
         app.create_task(self._success_rate_loop())

    def wire_handlers_and_run(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("startbot", self.cmd_startbot))
        self.app.add_handler(CommandHandler("stopbot", self.cmd_stopbot))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("history", self.cmd_history))
        self.app.add_handler(CommandHandler("approve", self.cmd_approve))
        self.app.add_handler(CommandHandler("unstuck", self.cmd_unstuck))
        self.app.add_handler(CommandHandler("setapproval", self.cmd_setapproval))
        self.app.add_handler(CommandHandler("counts", self.cmd_counts))
        self.app.add_handler(CommandHandler("queue", self.cmd_queue))
        self.app.add_handler(CommandHandler("successrate", self.cmd_successrate))  # <-- add this
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_fix_reply))
        self.app.add_handler(CallbackQueryHandler(self.on_cb_approve, pattern=r"^approve:"))
        self.app.add_handler(CallbackQueryHandler(self.on_cb_skip,    pattern=r"^skip:"))
        self.app.add_handler(CallbackQueryHandler(self.on_cb_details, pattern=r"^details:"))

        self.app.run_polling(drop_pending_updates=True, stop_signals=None)

if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing")
    bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, p2p_bot_service, redis_client)
    p2p_bot_service.telegram_bot = bot  # allow backend to send Telegram notifications
    bot.wire_handlers_and_run()

