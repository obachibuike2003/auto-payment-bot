# Bybit P2P Automation Bot

An automated bot that monitors Bybit P2P orders, verifies seller payment details, executes bank transfers via the Nomba API, and confirms payments back on Bybit — all with Telegram-based controls and notifications.

---

## Overview

When a buyer places a P2P order on Bybit, this bot:

1. Detects the pending order via polling
2. Extracts seller bank details (with AI fallback for messy data)
3. Verifies the account via Nomba's bank lookup
4. Matches the seller name against the resolved account name
5. Transfers funds via Nomba
6. Confirms the payment on Bybit
7. Notifies the operator via Telegram throughout the process

---

## Architecture

```
┌─────────────────────────────────┐
│         Flask Backend           │  app.py
│  - Bybit P2P polling & signing  │
│  - Nomba transfer execution     │
│  - Redis state management       │
│  - APScheduler job loop         │
│  - REST API for Telegram/UI     │
└────────────┬────────────────────┘
             │
    ┌────────▼────────┐     ┌──────────────┐
    │  Redis (state)  │     │  Telegram Bot │  telegram_runner.py
    └─────────────────┘     │  - Commands   │
                            │  - Callbacks  │
                            │  - Alerts     │
                            └──────────────┘
```

**Key components:**

- `app.py` — Flask app, bot service, Bybit and Nomba API clients, scheduler, webhooks
- `telegram_runner.py` — Telegram bot for operator controls and notifications

---

## Prerequisites

- Python 3.9+
- Redis (running locally or via a managed service)
- Bybit account with P2P API access
- Nomba business account with API credentials
- Telegram bot token (from [@BotFather](https://t.me/BotFather))
- OpenAI API key (for GPT name matching and bank resolution fallback)
- Groq API key (for AI payment detail extraction fallback)

---

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-directory>

# Install Python dependencies
pip install flask flask-apscheduler flask-cors python-dotenv redis requests \
            openai thefuzz python-telegram-bot pytz

# Start Redis
redis-server
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Bybit
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
BYBIT_BASE_URL=https://api.bybit.com

# Nomba
NOMBA_CLIENT_ID=your_nomba_client_id
NOMBA_CLIENT_SECRET=your_nomba_client_secret
NOMBA_ACCOUNT_ID=your_nomba_account_id
NOMBA_BASE_URL=https://api.nomba.com
NOMBA_SENDER_NAME=Your Business Name

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=2

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# AI
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key

# Bot behavior
POLLING_INTERVAL_SECONDS=5
MAX_ORDERS_PER_CYCLE=100
USE_NOMBA_FOR_TRANSFERS=true
ALLOW_WALLET_NO_LOOKUP=false
BACKEND_URL=http://127.0.0.1:5000
TELEGRAM_SUCCESS_RATE_EVERY_MIN=0   # 0 = disable auto-reports
```

---

## Running the Bot

Start the Flask backend and the Telegram bot in separate terminals:

```bash
# Terminal 1 — Flask backend
python app.py

# Terminal 2 — Telegram bot
python telegram_runner.py
```

The scheduler starts automatically when the backend receives its first HTTP request (or immediately on boot). The bot will begin polling Bybit for pending orders at the configured interval.

---

## Telegram Commands

| Command | Description |
|---|---|
| `/start` | Show available commands |
| `/startbot` | Start the order processing scheduler |
| `/stopbot` | Stop the scheduler (without killing the process) |
| `/status` | Show scheduler state, last cycle time, and settings |
| `/history [n]` | Show last `n` completed transfers (default 10) |
| `/queue` | List current pending Bybit orders with inline approve/skip buttons |
| `/approve <order_id>` | Manually approve an order for processing |
| `/unstuck <order_id>` | Remove an order from all stuck/failed sets so it retries |
| `/setapproval on\|off` | Toggle approval mode (require manual approval before each transfer) |
| `/counts` | Show counts of processed, stuck, and pending orders |
| `/successrate` | Show overall success rate statistics |

When an order gets stuck, the bot sends a notification with the order details. You can reply with corrected details in the format below to retry:

```
Bank: Access Bank
Account: 0123456789
Name: John Doe
```

---

## REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/control/start` | Start the scheduler |
| `POST` | `/control/stop` | Stop the scheduler |
| `GET` | `/control/status` | Get scheduler and bot status |
| `GET` | `/control/success-rate` | Get transfer success statistics |
| `POST` | `/webhook/nomba/transfer` | Nomba transfer status webhook |

---

## Order Processing Flow

```
Pending order detected
        │
        ▼
Already processed / stuck / cancelled?  ──► Skip
        │ No
        ▼
Fetch full order details from Bybit
        │
        ▼
Is order in a payable state?  ──► Skip (appealed / cancelled / paid)
        │ Yes
        ▼
Extract payment details (bank, account, name)
  └─ AI fallback (Groq LLM) if standard extraction fails
        │
        ▼
Approval mode enabled?  ──► Wait for /approve via Telegram
        │ No / Approved
        ▼
Resolve bank code (static map → fuzzy match → GPT fallback)
        │
        ▼
Check Nomba wallet balance
        │ Sufficient
        ▼
Resolve account name via Nomba bank lookup
        │
        ▼
Name match check (difflib → fuzzy → GPT)
        │ Passed
        ▼
Initiate Nomba transfer
        │
        ▼
Poll for transfer status (202 → retry up to 5x)
        │ Successful
        ▼
Confirm order as paid on Bybit
        │
        ▼
Send Bybit chat message + Telegram success notification
```

---

## Redis Keys Reference

| Key | Type | Description |
|---|---|---|
| `p2p_bot:processed_orders` | Set | Orders successfully completed |
| `p2p_bot:stuck_orders` | Set | Orders that failed and need intervention |
| `p2p_bot:insufficient_funds_orders` | Set | Orders skipped due to low balance |
| `p2p_bot:pending_transfers` | Set | Transfers initiated but not yet confirmed |
| `p2p_bot:cancelled_by_user_orders` | Set | Orders manually skipped |
| `p2p_bot:approved_orders` | Set | Orders approved in approval mode |
| `p2p_bot:transfers` | List | History of completed transfers (JSON) |
| `p2p_bot:order_details:<id>` | Hash | Cached order details (amount, bank, account, name) |
| `p2p_bot:pending_nomba_refs` | Hash | `order_id` → `merchant_tx_ref` mapping |
| `p2p_bot:nomba_tx_ref_to_order` | Hash | Reverse mapping for webhook lookups |
| `p2p_bot:approval_mode` | String | `true` or `false` |
| `p2p_bot:last_cycle_time` | String | Unix timestamp of last processing cycle |
| `p2p_bot:lock` | String | Distributed lock to prevent concurrent cycles |

---

## Safety Features

- **Distributed lock** — prevents multiple concurrent processing cycles
- **Double-payment guard** — checks `processed_orders` and `pending_transfers` before any transfer
- **Payable-state check** — skips orders that are appealed, paid, completed, or cancelled
- **Name verification** — multi-stage match (difflib → fuzzy → GPT) before sending funds
- **Invalid keyword detection** — rejects accounts containing phrases like "check dm" or "whatsapp"
- **Account number validation** — enforces 10-digit NUBAN format
- **Balance check** — aborts if Nomba wallet balance is insufficient
- **Idempotency keys** — Nomba transfers use `BYBIT_<order_id>` as the idempotency key to prevent duplicate debits

---

## Notes

- Wallet/PSB banks (OPay, PalmPay, 9PSB, SmartCash, MoMo) skip the account name lookup step since their APIs don't support it.
- The Groq AI fallback for payment detail extraction handles cases where sellers put account numbers in the wrong fields or use misspelled bank names.
- GPT (via OpenAI) is used as a last resort for both bank code resolution and name matching.
- Transfer amounts are rounded down to whole naira (kobo is stripped) before sending.
