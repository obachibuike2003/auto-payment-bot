import os
import time
import hmac
import hashlib
import json
import logging
from datetime import datetime, timedelta
import requests
from flask import Flask, request, jsonify, g
from flask_apscheduler import APScheduler
from flask_cors import CORS
from dotenv import load_dotenv
import redis
import hmac
import hashlib
import time
import requests
from collections import OrderedDict
from typing import Tuple, List, Union
import sys
import io
import difflib
from typing import Optional
from thefuzz import fuzz
from thefuzz import process
from threading import Thread
import pytz
from decimal import Decimal, ROUND_DOWN
import uuid
from openai import OpenAI



       

# Load environment variables from .env file
load_dotenv()

# The client will now automatically find the key



import re
import unicodedata

BRANCH_NOISE = {
    "branch", "br.", "br", "office", "market", "shop", "road", "rd", "street",
    "st", "avenue", "ave", "lagos", "abuja", "ibadan", "state", "nigeria"
}

def normalize_account_number(acc: str) -> str:
    """Keep digits only (removes spaces, dashes, etc.)."""
    return re.sub(r"\D", "", str(acc or ""))

def _clean_text(s: str) -> str:
    if not s:
        return ""
    # normalize unicode, collapse spaces, strip punctuation at ends
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def looks_like_person_name(text: str) -> bool:
    """
    Heuristic: at least two alphabetic tokens, no long non-letters,
    and not dominated by branch/location words.
    """
    if not text:
        return False
    t = _clean_text(text)
    tokens = [tok for tok in re.split(r"[^\w]+", t) if tok]
    if len(tokens) < 2:
        return False
    alpha_tokens = [tok for tok in tokens if re.fullmatch(r"[A-Za-z]+", tok)]
    if len(alpha_tokens) < 2:
        return False
    low = t.lower()
    noise_hits = sum(1 for w in BRANCH_NOISE if w in low)
    # allow a few noise words, but not if it's clearly an address/branch line
    return noise_hits <= 1

def normalize_person_name(text: str) -> str:
    """Title-case and trim multiple spaces."""
    t = _clean_text(text)
    return t.title()

def get_best_seller_name(order_result: dict) -> str:
    """
    Look for a plausible seller name across multiple fields.
    Priority:
      1) result['sellerRealName']
      2) paymentTermList fields: accountName, realName
      3) 'branch' style fields: bankBranch/branch/branchName/remark/note/memo (if they look like names)
    """
    # 1) direct field
    direct = order_result.get("sellerRealName") or order_result.get("seller_name")
    direct = normalize_person_name(direct) if direct else ""
    if looks_like_person_name(direct):
        return direct

    # 2) scan payment term list
    for term in order_result.get("paymentTermList", []) or []:
        for k in ("accountName", "realName", "holderName", "name"):
            cand = normalize_person_name(term.get(k))
            if looks_like_person_name(cand):
                return cand

    # 3) last resort: branch-ish fields sometimes contain the name
    for k in ("bankBranch", "branch", "branchName", "remark", "note", "memo"):
        cand = normalize_person_name(order_result.get(k))
        if looks_like_person_name(cand):
            return cand

    # If still nothing, return whatever best we had (even if empty)
    return direct or ""



# Configure your logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gpt_bank_name_to_code(user_bank_name: str, bank_map: dict) -> Optional[Tuple[str, str]]:
    """Use GPT to intelligently match a user's bank name to a real bank name + code."""
    if not client:
        logger.error("OpenAI client not initialized.")
        return None

    bank_list = "\n".join([f"- {name} (Code: {code})" for name, code in bank_map.items()])
    prompt = f"""
    The user typed the bank name: "{user_bank_name}"

    Below is a list of valid Nigerian banks and their codes:
    {bank_list}

    From the list above, which bank do you think they meant?

    Just return the correct bank name and code in this format: BANK NAME | CODE

    If you can't find any match, return "Unknown".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that matches bank names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        result = response.choices[0].message.content.strip()
        if '|' in result:
            bank_name, bank_code = result.split(' | ')
            return bank_name, bank_code
        return None
    except Exception as e:
        logger.error(f"GPT bank name lookup failed: {e}")
        return None
    
# --- The rest of your code is fine as it defines `prompt` inside its function. ---

def gpt_fallback_name_match(name1: str, name2: str) -> bool:
    """
    Uses GPT to determine if two names likely refer to the same person.
    It's instructed to be non-strict and approve if there's a 50% or more similarity.
    """
    if not client:
        logger.error("OpenAI client not initialized.")
        return False

    prompt = f"""
    Are these two names likely the same person? Be very lenient. If one name from the seller's name matches a name from the lookup, or if the names are at least 50% similar, approve it.

    Name 1 (Seller): {name1}
    Name 2 (Account Lookup): {name2}

    Reply only with "YES" or "NO".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = response.choices[0].message.content.strip().lower()
        return "yes" in result
    except Exception as e:
        logger.error(f"GPT name match failed: {e}")
        return False


def process_name_match(order_id, seller_bank_name, seller_real_name, resolved_account_name, wallet_keywords):
    """
    Use GPT to match names for banks. GPT is FIRST priority.
    """
    is_wallet = any(word in seller_bank_name.lower() for word in wallet_keywords)

    if not is_wallet:
        # ✅ Step 1: Use GPT to compare names
        gpt_ok = gpt_fallback_name_match(seller_real_name, resolved_account_name)
        if gpt_ok:
            logger.info(f"✅ GPT accepted name match for order {order_id}")
            return True
        
        # ❌ GPT said no → log detailed info and fail
        logger.warning(f"❌ GPT rejected name match for order {order_id}: '{seller_real_name}' vs '{resolved_account_name}'")
        return False

        # Wallet banks skip strict name match
        logger.info(f"⚠️ Skipping name match for WALLET ({seller_bank_name})")
        return True
def compare_names(name1, name2):
    """
    Compares two names for similarity using difflib.
    Returns a tuple: (match: bool, score: float, overlap: int, debug_info: dict)
    """
    if not name1 or not name2:
        return False, 0.0, 0, {"reason": "One or both names are empty"}

    clean1 = "".join(name1.lower().split())
    clean2 = "".join(name2.lower().split())

    similarity = difflib.SequenceMatcher(None, clean1, clean2).ratio()
    overlap = len(set(clean1) & set(clean2))
    debug_info = {
        "clean1": clean1,
        "clean2": clean2,
        "similarity": similarity,
        "overlap": overlap
    }

    return similarity >= 0.8, similarity, overlap, debug_info

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def fallback_lookup(order_id, seller_real_name, resolved_account_name, logger):
    """
    Fallback method if name comparison fails.
    Currently always returns False but can be extended.
    """
    logger.warning(f"Fallback lookup triggered for Order {order_id}: "
                   f"Seller='{seller_real_name}' vs Resolved='{resolved_account_name}'")
    return False


sys.stdout.reconfigure(encoding='utf-8')


# Load environment variables from .env file
load_dotenv()

# --- Flask App Setup (Moved to top) ---
app = Flask(__name__)
app.config['SCHEDULER_TIMEZONE'] = 'Africa/Lagos'  # Set timezone to Lagos
CORS(app) # Enable CORS for your Flask app

# --- Configuration ---
# Bybit API Credentials
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
# IMPORTANT: Bybit P2P API does NOT use /v5/ prefix in its base path
BYBIT_BASE_URL = os.getenv('BYBIT_BASE_URL', "https://api.bybit.com") # Default to production if not set

# Paystack API Credentials
PAYSTACK_SECRET_KEY = os.getenv('PAYSTACK_SECRET_KEY')
PAYSTACK_BASE_URL = "https://api.paystack.co"

# Nomba API Credentials
NOMBA_CLIENT_ID = os.getenv('NOMBA_CLIENT_ID')
NOMBA_CLIENT_SECRET = os.getenv('NOMBA_CLIENT_SECRET')
NOMBA_ACCOUNT_ID = os.getenv('NOMBA_ACCOUNT_ID') # Your Business ID from Nomba dashboard
NOMBA_BASE_URL = os.getenv('NOMBA_BASE_URL') # e.g., https://sandbox.nomba.com or https://api.nomba.com
NOMBA_SENDER_NAME = os.getenv('NOMBA_SENDER_NAME') # NEW: sender name

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 2))



# Bot Settings
POLLING_INTERVAL_SECONDS = int(os.getenv('POLLING_INTERVAL_SECONDS', 5))
MAX_ORDERS_PER_CYCLE = int(os.getenv('MAX_ORDERS_PER_CYCLE', 100))
APPROVAL_MODE_ENABLED_DEFAULT = False

# Wallet lookup fallback settings
WALLET_NO_LOOKUP_MAX = 500000  # Maximum amount allowed for wallet fallback (adjust as needed)
WALLET_NO_LOOKUP_NAME_SCORE = 80  # Minimum name similarity score for wallet fallback

# Feature Toggles
USE_NOMBA_FOR_TRANSFERS = os.getenv('USE_NOMBA_FOR_TRANSFERS', 'false').lower() == 'true'
ALLOW_WALLET_NO_LOOKUP = os.getenv('ALLOW_WALLET_NO_LOOKUP', 'false').lower() == 'true'  # Add this line

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("p2p_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # Use sys.stdout, which you already set to UTF-8
    ]
)
logger = logging.getLogger('P2P_Bot_Backend')

# --- Redis Client Initialization ---
try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.critical(f"Could not connect to Redis: {e}. The bot will not function correctly without Redis.")
    redis_client = None


    
def _safe_fire_and_forget(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)



# --- Bybit Client (P2P Specific) ---
class BybitP2PClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        if not self.api_key or not self.api_secret:
            logger.warning("Bybit API Key or Secret is missing. Bybit API calls will likely fail.")
        logger.info(f"Initializing Bybit P2P client with API Key ending in: {api_key[-4:] if api_key else 'N/A'}")

    def _generate_bybit_signature_for_post(self, payload: dict, timestamp: str, recv_window: str) -> Tuple[str, str]:
        """Generates Bybit signature for POST requests with alphabetically sorted payload."""
        sorted_payload_str = json.dumps(OrderedDict(sorted(payload.items())), separators=(',', ':'))
        logger.debug(f"JSON Payload String for Signature (ORDERED & COMPACT): '{sorted_payload_str}'")
        
        sign_str = timestamp + self.api_key + recv_window + sorted_payload_str
        
        logger.debug(f"Signature origin string: '{sign_str}'")

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        logger.debug(f"Generated signature: {signature}")
        return signature, sorted_payload_str

    # No longer needed for order/info as it's POST, but kept for potential future GET requests
    def _generate_bybit_signature_for_get(self, params: dict, timestamp: str, recv_window: str) -> Tuple[str, str]:
        """Generates Bybit signature for GET requests with alphabetically sorted query parameters."""
        # Sort query parameters alphabetically and form query string
        sorted_params_str = '&'.join(f"{key}={value}" for key, value in sorted(params.items()))
        logger.debug(f"Query String for Signature (ORDERED): '{sorted_params_str}'")

        sign_str = timestamp + self.api_key + recv_window + sorted_params_str
        logger.debug(f"Signature origin string: '{sign_str}'")

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        logger.debug(f"Generated signature: {signature}")
        return signature, sorted_params_str # Return sorted_params_str to be appended to URL

    def get_pending_orders(self) -> List[dict]:
        """
        Fetches a list of pending Bybit P2P orders and then
        fetches detailed payment information for each one to populate the dashboard.
        """
        if not self.api_key or not self.api_secret:
            logger.error("Bybit API keys are not configured. Cannot fetch pending orders.")
            return []

        url = f"{self.base_url}/v5/p2p/order/pending/simplifyList"
        
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        payload = {
            "page": 1,
            "size": 30,
        }

        signature, payload_str_for_request = self._generate_bybit_signature_for_post(payload, timestamp, recv_window)
        
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"Attempting to fetch pending Bybit orders via POST to {url}. Sending payload string: '{payload_str_for_request}'")
            response = requests.post(url, headers=headers, data=payload_str_for_request, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Bybit API Raw Response (simplifyList): {data}")

            if data and data.get('ret_code') == 0 and data.get('result'):
                orders = []
                order_list = data['result'].get('list', data['result'].get('items', []))

                for order in order_list:
                    order_id = order.get('id', order.get('orderNo', 'N/A'))
                    
                    # Fetch full payment details for the dashboard display
                    payment_info = self.get_order_details(order_id)
                    
                    # --- Mapping Bybit API keys to frontend expected keys ---
                    seller_info = {
                        'bankName': payment_info.get('bankName', 'N/A') if payment_info else 'N/A',
                        'bankAccountNo': payment_info.get('accountNo', 'N/A') if payment_info else 'N/A',
                        'accountHolderName': payment_info.get('realName', 'N/A') if payment_info else 'N/A',
                    }
                    # --- END Mapping ---

                    status_raw = order.get('status')
                    status_str = 'Unknown'
                    if status_raw == 0:
                        status_str = 'Unpaid'
                    elif status_raw == 1:
                        status_str = 'Paid'
                    elif status_raw == 2:
                        status_str = 'Completed'
                    elif status_raw == 3:
                        status_str = 'Cancelled'
                    elif status_raw == 10: # Status 10 is typically 'PENDING PAYMENT'
                        status_str = 'Pending Payment'
                    else:
                        status_str = str(status_raw)

                    orders.append({
                        'orderId': order_id,
                        'fiatAmount': float(order.get('amount', 0.0)), # Correctly using 'amount'
                        'sellerInfo': seller_info, # Now populating with actual details or N/A if fetch failed
                        'status': status_str,
                        'createdAt': order.get('createDate', order.get('createdTime', 'N/A')),
                    })
                logger.info(f"Found {len(orders)} pending Bybit orders (with payment details) for dashboard.")
                return orders
            else:
                logger.warning(f"Bybit API returned non-success ret_code {data.get('ret_code')}: {data.get('ret_msg')}. Full response: {data}")
                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching pending Bybit orders from simplifyList: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Bybit API error response content: {e.response.text}")
            return []

    def get_order_details(self, order_id: str) -> Union[dict, None]:
        """
        Fetches full details for a specific Bybit P2P order, including payment methods.
        Uses /v5/p2p/order/info endpoint with POST request and 'orderId' parameter.
        """
        if not self.api_key or not self.api_secret:
            logger.error("Bybit API keys are not configured. Cannot fetch order details.")
            return None

        endpoint = "/v5/p2p/order/info"
        
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        payload = {
            "orderId": order_id
        }

        signature, payload_str_for_request = self._generate_bybit_signature_for_post(payload, timestamp, recv_window)
        
        url = f"{self.base_url}{endpoint}"

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"Attempting to fetch Bybit order details for {order_id} via POST to {url}. Sending payload string: '{payload_str_for_request}'")
            response = requests.post(url, headers=headers, data=payload_str_for_request, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Bybit API Raw Response (order/info for {order_id}): {data}")
            return data  # Return the full response, not just paymentTermList
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Bybit order details for {order_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Bybit API error response content: {e.response.text}")
            return None
        
        
    

    def confirm_p2p_order_paid(self, order_id: str, payment_type: str, payment_id: str) -> dict:
        """
        Mark order as PAID on Bybit P2P.
        POST /v5/p2p/order/pay
        """
        ts = str(int(time.time() * 1000))
        recv_window = "50000"

        payload = {
            "orderId": str(order_id),
            "paymentType": str(payment_type),  # must be string
            "paymentId": str(payment_id),      # must be string
        }

        # Generate signature for POST request
        signature, payload_str_for_request = self._generate_bybit_signature_for_post(payload, ts, recv_window)
        url = f"{self.base_url}/v5/p2p/order/pay"
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(url, headers=headers, data=payload_str_for_request, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error confirming Bybit P2P order as paid: {e}")
            return {"ret_code": -1, "ret_msg": str(e)}

    def send_p2p_chat_message(self, order_id: str, message: str, content_type: str = "str") -> dict:
        """
        Send a chat message inside a Bybit P2P order.
        content_type: str | pic | pdf | video
        """
        endpoint = "/v5/p2p/order/message/send"
        url = f"{self.base_url}{endpoint}"

        ts = str(int(time.time() * 1000))
        recv_window = "5000"

        payload = {
            "orderId": str(order_id),
            "message": message,
            "contentType": content_type,
            "msgUuid": uuid.uuid4().hex
        }

        signature, payload_str = self._generate_bybit_signature_for_post(
            payload, ts, recv_window
        )

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(url, headers=headers, data=payload_str, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to send Bybit chat message for {order_id}: {e}")
            return {"ret_code": -1, "ret_msg": str(e)}
   

# Nomba API Class (Improved with balance check and proper sender name handling)
class NombaAPI:
    def __init__(self, client_id, client_secret, account_id, base_url, sender_name):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = account_id
        self.base_url = base_url
        self.sender_name = sender_name # Store sender name
        self.access_token = None
        self.token_expiry = 0 # Unix timestamp
        logger.info("NombaAPI initialized.")

    def _authenticate(self):
        # Check if token is still valid (refresh 60 seconds before expiry)
        if self.access_token and self.token_expiry > time.time() + 60:
            logger.debug("Using cached Nomba access token.")
            return True

        endpoint = "/v1/auth/token/issue"
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json', 'accountId': self.account_id}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        logger.info("Attempting to authenticate with Nomba API.")
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            response_json = response.json()

            if response_json.get('code') == '00' and response_json.get('data', {}).get('access_token'):
                self.access_token = response_json['data']['access_token']
                expires_at_str = response_json['data']['expiresAt']
                expires_at_dt = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                self.token_expiry = expires_at_dt.timestamp()
                logger.info("Successfully authenticated with Nomba API.")
                return True
            else:
                logger.error(f"Nomba Authentication failed: {response_json.get('description', 'Unknown error')} - {response_json.get('message', '')}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Nomba Authentication network error: {e}")
            return False

    def _send_request(self, method, endpoint, data=None, requires_auth=True, idempotency_key=None):
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json', 'accountId': self.account_id}

        if requires_auth:
            if not self._authenticate():
                logger.error("Failed to authenticate with Nomba, cannot send request.")
                return {"code": "99", "description": "Authentication failed"} # Custom error code

            headers["Authorization"] = f"Bearer {self.access_token}"
        
        if idempotency_key:
           headers['X-Idempotency-Key'] = idempotency_key

        try:
            response = requests.request(method, url, headers=headers, json=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Nomba API request to {endpoint} timed out after 300 seconds.")
            return {"code": "97", "description": "Time Out Waiting For Response"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Nomba API request to {endpoint} failed: {e}")
            if e.response is not None:
                logger.error(f"Nomba API Error Response: {e.response.text}")
                try:
                    return e.response.json()
                except json.JSONDecodeError:
                    return {"code": "99", "description": f"Non-JSON error response from Nomba: {e.response.text}"}
            return {"code": "99", "description": f"Request failed: {e}"}

    def resolve_bank_account(self, account_number, bank_code):
        endpoint = "/v1/transfers/bank/lookup"
        data = {
            "accountNumber": account_number,
            "bankCode": bank_code
        }
        logger.info(f"Performing Nomba account lookup for {account_number} at bank {bank_code}")
        return self._send_request('POST', endpoint, data)

    def initiate_fund_transfer(self, amount, account_number, account_name, bank_code, merchant_tx_ref, narration=None):
        endpoint = "/v1/transfers/bank"
        data = {
            "amount": amount,
            "accountNumber": account_number,
            "accountName": account_name,
            "bankCode": bank_code,
            "merchantTxRef": merchant_tx_ref,
            "senderName": self.sender_name # Use the sender_name from initialization
        }
        if narration:
            data["narration"] = narration
        
        logger.info(f"Initiating Nomba fund transfer for {amount} NGN to {account_name} ({account_number})")
        return self._send_request('POST', endpoint, data, idempotency_key=merchant_tx_ref)

# ... (other methods like _authenticate, _send_request, etc.)

    def get_wallet_balance(self):
        """
        Fetch available wallet balance from Nomba API using the correct endpoint.
        """
        if not self._authenticate():
            logger.error("Failed to authenticate before fetching wallet balance.")
            return 0.0

        url = f"{self.base_url}/v1/accounts/balance"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "accountId": self.account_id
        }

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            logger.info(f"Nomba raw balance API response: {data}")

            if data.get("code") == "00":
                # FIX: Use 'amount' instead of 'availableBalance'
                balance = float(data.get('data', {}).get('amount', 0.0))
                logger.info(f"Nomba wallet balance retrieved: {balance} NGN")
                return balance
            else:
                logger.error(f"Failed to fetch balance: {data.get('description')}")
                return 0.0

        except Exception as e:
            logger.error(f"Failed to retrieve Nomba wallet balance: {e}. Defaulting to 0.")
            return 0.0
    # --- Function to fetch Nomba bank codes dynamically ---
    def check_transfer_status(self, merchant_tx_ref):
        url = f"https://api.nomba.com/v1/transactions/requery/{merchant_tx_ref}"
        headers = {
            "accountId": self.account_id,         # <-- set this in __init__!
            "Authorization": f"Bearer {self.access_token}" # <-- set this in __init__!
        }
        response = requests.get(url, headers=headers)
        try:
            return response.json()
        except Exception as e:
            return {"code": "error", "description": str(e), "data": {}}
     
def get_nomba_banks(nomba_api):
    if not nomba_api._authenticate():
        logger.error("Could not authenticate with Nomba to fetch bank list.")
        return {}

    url = f"{nomba_api.base_url}/v1/transfers/banks"
    headers = {
        "Authorization": f"Bearer {nomba_api.access_token}",
        "accountId": nomba_api.account_id,  # <-- Add this line
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") is True and "data" in data:
            # Create dictionary: normalized bank name => bank code
            bank_codes = {}
            for bank in data["data"]:
                normalized_name = bank["name"].lower().replace(" ", "")
                bank_codes[normalized_name] = bank["code"]

            logger.info(f"Loaded {len(bank_codes)} bank codes from Nomba.")
            return bank_codes
        else:
            logger.error(f"Unexpected bank list response: {data}")
            return {}

    except Exception as e:
        logger.error(f"Error fetching Nomba bank list: {e}")
        return {}




class P2PBotService:
    def __init__(self, bybit_api, nomba_api, redis_client,telegram_bot=None):
        self.bybit_api = bybit_api
        self.nomba_api = nomba_api
        self.redis_client = redis_client
        self.telegram_bot = telegram_bot  # Store telegram bot instance
        self.is_processing = False

        self.use_approval_mode = self.redis_client.get('p2p_bot:approval_mode')
        if self.use_approval_mode is None:
            self.use_approval_mode = APPROVAL_MODE_ENABLED_DEFAULT
            self.redis_client.set('p2p_bot:approval_mode', 'true' if self.use_approval_mode else 'false')
            logger.info(f"Initialized approval mode to default: {self.use_approval_mode}")
        else:
            self.use_approval_mode = self.use_approval_mode.lower() == 'true'
            logger.info(f"Loaded approval mode from Redis: {self.use_approval_mode}")

        logger.info("P2PBotService initialized.")

    def _rget(self, key, field, default='N/A'):
        """Helper to safely get Redis hash values and decode bytes"""
        v = self.redis_client.hget(key, field)
        if isinstance(v, bytes):
            v = v.decode('utf-8', 'ignore')
        return v if v not in (None, '') else default
    
    def _handle_transfer_failure(self, order_id, message, log_level='error'):
        """Log, classify, clear in-progress flags, persist details, and notify via Telegram."""

    # 1) Clear any 'in-progress' state so future cycles can try again
        try:
          if self.redis_client:
            # remove from pending set
            self.redis_client.srem("p2p_bot:pending_transfers", order_id)

            # get and clear nomba refs (both directions)
            ref = self.redis_client.hget("p2p_bot:pending_nomba_refs", order_id)
            self.redis_client.hdel("p2p_bot:pending_nomba_refs", order_id)
            if ref:
                self.redis_client.hdel("p2p_bot:nomba_tx_ref_to_order", ref)
        except Exception as e:
          logger.error(f"Failed to clear pending flags for {order_id}: {e}")

    # 2) Log with appropriate level (keep your behavior)
        if log_level == 'critical':
         logger.critical(message)
        elif log_level == 'warning':
         logger.warning(message)
        elif log_level == 'info':
         logger.info(message)
        else:
         logger.error(message)

    # 3) Classify failure (keep your behavior)
        msg_low = (message or "").lower()
        if any(w in msg_low for w in ["insufficient", "low balance", "balance too low"]):
         if self.redis_client:
            self.redis_client.sadd("p2p_bot:insufficient_funds_orders", order_id)
        else:
         if self.redis_client:
            self.redis_client.sadd("p2p_bot:stuck_orders", order_id)

    # 4) Build order details from Redis (keep your behavior)
        try:
            order_details = {
                'order_id': order_id,
                'reason': message,
                'amount': self._rget(f'p2p_bot:order_details:{order_id}', 'amount'),
                'seller_bank_name': self._rget(f'p2p_bot:order_details:{order_id}', 'bank'),
                'seller_account_no': self._rget(f'p2p_bot:order_details:{order_id}', 'account'),
                'seller_real_name': self._rget(f'p2p_bot:order_details:{order_id}', 'name'),
        }
        except Exception:
            order_details = {'order_id': order_id, 'reason': message}

    # 5) Notify Telegram with your editable template (keep your behavior)
        tg = getattr(self, "telegram_bot", None)
        if tg:
            try:
                _safe_fire_and_forget(
                    tg.send_stuck_order_notification(order_id, message, order_details)
                )
            except Exception as e:
                logger.error(f"Failed to schedule Telegram notify for {order_id}: {e}")
    def _get_clean_bank(self, raw_input):
        if not raw_input:
            return ""
        # The noise list should be defined here or as a class constant
        noise = ["bank", "mfb", "microfinance", "paymentservice", "psb", "microfinace", "digital", "ltd", "limited"]
        clean = str(raw_input).lower()
        for word in noise:
            clean = clean.replace(word, " ")
        return clean.strip()

    def _extract_payment_details(self, order_details, order_id):
        payment_terms = order_details.get('paymentTermList') or []
        if not payment_terms:
            self._handle_transfer_failure(order_id, "No payment terms found", log_level="warning")
            return None, None, None

        # Prefer Bank Transfer (Type 14) or fallback to first available
        term = next((t for t in payment_terms if str(t.get('paymentType')) == '14'), payment_terms[0])

        known_banks = [
            "uba", "access", "firstbank", "gtbank", "guarantytrust", "zenith", "kuda", "opay", "unitedbankofafrica",
            "palmpay", "moniepoint", "stanbic", "wema", "fidelity", "fcmb", "9japay", "jaizbank",
            "union", "sterling", "ecobanknigeria", "9psb", "fairmoney", "providus","paga",
            "diamond", "keystone", "polaris", "paycom(opay)", "standbicibtc", "rubies", "sparkle", "vfd", "kekabank"
        ]

        # --- 1. FIND THE 10-DIGIT ACCOUNT NUMBER ---
        seller_account_no = ""
        acct_field_used = ""

        val = normalize_account_number(term.get('accountNo', ''))
        if len(val) == 10:
            seller_account_no = val
            acct_field_used = 'accountNo'
        else:
            for field in ['debitCardNumber', 'branchName', 'bankName']:
                val = normalize_account_number(term.get(field, ''))
                if len(val) == 10:
                    seller_account_no = val
                    acct_field_used = field
                    break

        # --- 2. FIND THE BANK NAME (HIERARCHICAL SEARCH) ---
        seller_bank_name = ""
        
        # Priority A: Check the 'bankName' field first
        bank_raw = term.get('bankName', '')
        bank_clean = self._get_clean_bank(bank_raw)
        
        for bank in known_banks:
            if bank in bank_clean:
                seller_bank_name = bank
                break

        # Priority B: If Bank Name was empty/garbage (like "APP"), check 'branchName'
        if not seller_bank_name:
            branch_raw = term.get('branchName', '')
            branch_clean = self._get_clean_bank(branch_raw)
            for bank in known_banks:
                if bank in branch_clean:
                    seller_bank_name = bank
                    break
        else:
            branch_clean = "" # Initialize for the fuzzy logic below if bank was found

        # Priority C: Last Resort - Fuzzy Match (handles typos like "Kudda")
        if not seller_bank_name:
            import difflib
            # Use whichever field actually had text in it
            best_input = bank_clean if bank_clean else (branch_clean if 'branch_clean' in locals() else "")
            if best_input:
                matches = difflib.get_close_matches(best_input, known_banks, n=1, cutoff=0.7)
                if matches:
                    seller_bank_name = matches[0]

        # --- 3. FIND THE REAL NAME ---
        seller_real_name = (term.get('realName') or order_details.get('sellerRealName') or '').strip()

        # Final Validation
        if not (seller_account_no and seller_bank_name and seller_real_name):
            # --- AI FALLBACK INITIATED ---
            print(f"Standard extraction failed for Order {order_id}. Consulting AI...")
            
            ai_data = self._ai_extraction_fallback(term, known_banks)
            
            if ai_data:
                seller_bank_name = seller_bank_name or ai_data.get('bank')
                seller_account_no = seller_account_no or ai_data.get('account_number')
                seller_real_name = seller_real_name or ai_data.get('name')

        # Final Check after AI intervention
        if not (seller_account_no and seller_bank_name and seller_real_name):
            error_msg = f"Incomplete details. Acct: {seller_account_no}, Bank: {seller_bank_name}, Name: {seller_real_name}"
            self._handle_transfer_failure(order_id, error_msg, log_level="warning")
            return None, None, None
        
        return seller_bank_name, seller_account_no, seller_real_name

    
    def _ai_extraction_fallback(self, raw_term_data, known_banks):
        """
        Uses Groq LLM to intelligently map messy or misspelled data 
        to your existing list of known banks.
        """
        client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        
        # Convert your list to a string for the AI to read
        valid_banks_str = ", ".join(known_banks)

        prompt = f"""
        You are a financial data parser. A user has provided messy payment details.
        
        YOUR GOAL:
        Extract the bank name, 10-digit account number, and account holder name.
        
        REFERENCE LIST OF VALID BANKS:
        {valid_banks_str}
        
        HUMAN-LIKE REASONING RULES:
        1. If the bank name is misspelled (e.g., 'Kudda' or 'Opai'), map it to the closest match in the REFERENCE LIST.
        2. If the fields are swapped (e.g., the account number is in the 'bankName' field), identify the 10-digit string and label it 'account_number'.
        3. If multiple banks are mentioned, pick the one that looks like the primary destination.
        4. Ignore 'APP' or garbage text.

        INPUT DATA:
        {raw_term_data}
        
        RETURN ONLY JSON:
        {{"bank": "match_from_list", "account_number": "10_digits", "name": "extracted_name"}}
        """
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise data extraction tool. Only output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            # AI fallback failed
            logger.error(f"AI extraction failed: {e}")
            return None
            return None

    def _store_order_details(self, order_id, amount, bank_name, account_no, real_name):
        """Store order details in Redis for later use in notifications"""
        if not self.redis_client:
            return
        self.redis_client.hset(f'p2p_bot:order_details:{order_id}', mapping={
            'account': normalize_account_number(account_no),
            'bank': bank_name or '',
            'account': str(account_no or ''),
            'name': real_name or '',
            'timestamp': str(time.time()),
        })
        self.redis_client.expire(f'p2p_bot:order_details:{order_id}', 86400)
        
    def set_approval_mode(self, enabled):
        self.use_approval_mode = enabled
        if self.redis_client:
            self.redis_client.set('p2p_bot:approval_mode', 'true' if enabled else 'false')
        logger.info(f"Approval mode set to: {self.use_approval_mode}")

    def approve_order(self, order_id):
        if self.redis_client:
            self.redis_client.sadd('p2p_bot:approved_orders', order_id)
            logger.info(f"Order {order_id} manually approved.")
            return {"status": "success", "message": f"Order {order_id} approved for processing."}
        return {"status": "error", "message": "Redis not connected. Cannot approve order."}

    def cancel_order_by_user(self, order_id):
        if self.redis_client:
            self.redis_client.sadd('p2p_bot:cancelled_by_user_orders', order_id)
            logger.info(f"Order {order_id} marked as cancelled by user.")
            return {"status": "success", "message": f"Order {order_id} marked as cancelled by user."}
        return {"status": "error", "message": "Redis not connected. Cannot cancel order."}

    def unstuck_order(self, order_id):
        if self.redis_client:
            self.redis_client.srem('p2p_bot:stuck_orders', order_id)
            self.redis_client.srem('p2p_bot:insufficient_funds_orders', order_id)
            self.redis_client.srem('p2p_bot:processed_orders', order_id)
            self.redis_client.srem('p2p_bot:cancelled_by_user_orders', order_id)
            self.redis_client.srem('p2p_bot:approved_orders', order_id)
            logger.info(f"Order {order_id} unstuck and removed from all tracking sets.")
            return {"status": "success", "message": f"Order {order_id} unstuck. It will be re-evaluated in the next cycle."}
        return {"status": "error", "message": "Redis not connected. Cannot unstuck order."}

    def cleanup_all_data(self):
        if self.redis_client:
            keys_to_delete = [
                'p2p_bot:stuck_orders',
                'p2p_bot:processed_orders',
                'p2p_bot:insufficient_funds_orders',
                'p2p_bot:transfers',
                'p2p_bot:paid_recipients',
                'p2p_bot:cancelled_by_user_orders',
                'p2p_bot:approved_orders',
                'p2p_bot:lock',
                'p2p_bot:approval_mode'
            ]
            for key in keys_to_delete:
                self.redis_client.delete(key)
            logger.info("All bot data in Redis has been cleaned up.")
            self.use_approval_mode = APPROVAL_MODE_ENABLED_DEFAULT
            self.redis_client.set('p2p_bot:approval_mode', 'true' if self.use_approval_mode else 'false')
            return {"status": "success", "message": "All bot data has been cleaned up."}
        return {"status": "error", "message": "Redis not connected. Cannot clean up data."}

    def get_bot_status(self):
        status = "unknown"
        message = "Bot status unknown."
        
        scheduler_active = False
        # Access app.apscheduler safely after it's initialized by before_request
        if hasattr(app, 'apscheduler') and app.apscheduler:
            scheduler_active = app.apscheduler.running
        else:
            message = "Scheduler not yet initialized or active (waiting for first request)."

        if not self.redis_client:
            status = "critical"
            message = "Redis is not connected. Bot cannot function."
            scheduler_active = False # If Redis is down, scheduler effectively cannot run bot logic

        elif self.redis_client.get('p2p_bot:lock') == 'locked':
            status = "running"
            message = "Bot is currently processing a cycle."
        elif scheduler_active:
            status = "running"
            status = "running"
            message = "Bot is currently processing a cycle."
        elif scheduler_active:
            status = "running"
            message = "Bot scheduler is active and waiting for next cycle."
        else:
            status = "paused"
            message = "Bot scheduler is paused. Click 'Start Automation' to resume."

        last_cycle_time = self.redis_client.get('p2p_bot:last_cycle_time') if self.redis_client else None

        return {
            "status": status,
            "message": message,
            "scheduler_active": scheduler_active,
            "last_cycle": float(last_cycle_time) if last_cycle_time else None,
            "use_approval_mode": self.use_approval_mode,
            "use_nomba_for_transfers": USE_NOMBA_FOR_TRANSFERS
        }

    def get_pending_orders_data(self):
        if not self.redis_client:
            return {"data": [], "message": "Redis not connected. Cannot fetch orders."}

        # Call get_p2p_orders with fixed parameters for side="Buy" and status="Pending"
        # Using page=1, limit=50 as per documentation
        pending_orders = self.bybit_api.get_pending_orders(limit=50, page=1)
        
        # Check for error response from BybitAPI._send_request using 'retCode'
        if not pending_orders or pending_orders('retCode') != 0:
            logger.error(f"Failed to fetch pending Bybit orders: {pending_orders.get('retMsg', 'Unknown error')}")
            # Return a structured error response that the frontend can handle
            return {"data": [], "message": pending_orders.get('retMsg', 'Failed to fetch pending orders from Bybit API.')}

        # Use 'items' for P2P API response parsing for order list
        pending_orders = pending_orders.get('result', {}).get('items', []) or []
        logger.info(f"Found {len(pending_orders)} pending Bybit orders.")

        dashboard_orders = []
        for order in pending_orders:
            order_id = order['id']
            
            is_processed = self.redis_client.sismember("p2p_bot:processed_orders", order_id)
            is_stuck = self.redis_client.sismember("p2p_bot:stuck_orders", order_id)
            is_insufficient_funds = self.redis_client.sismember("p2p_bot:insufficient_funds_orders", order_id)
            is_cancelled_by_user = self.redis_client.sismember("p2p_bot:cancelled_by_user_orders", order_id)
            is_approved = self.redis_client.sismember("p2p_bot:approved_orders", order_id)

            seller_info = {
                "bankName": "N/A",
                "bankAccountNo": "N/A",
                "accountHolderName": "N/A"
            }
            # The 'items' in simplifyList response has sellerRealName, but not full bank details.
            # We still need to call get_p2p_order_info for full details, but for the dashboard
            # we can show what's immediately available.
            seller_info['accountHolderName'] = order.get('sellerRealName', 'N/A')


            dashboard_orders.append({
                "orderId": order_id,
                "fiatAmount": order['amount'],
                "status": order['status'], # This will be a string like "PENDING"
                "createdAt": order['createDate'],
                "sellerInfo": seller_info,
                "isProcessed": is_processed,
                "isStuck": is_stuck,
                "isInsufficientFunds": is_insufficient_funds,
                "isCancelledByUser": is_cancelled_by_user,
                "isApproved": is_approved
            })
        return {"data": dashboard_orders, "message": "Pending orders fetched successfully."}

    def get_transfer_history_data(self):
        if not self.redis_client:
            return {"data": [], "message": "Redis not connected. Cannot fetch transfers."}

        transfers_json = self.redis_client.lrange('p2p_bot:transfers', 0, -1)
        transfers = [json.loads(t) for t in transfers_json]
        return {"data": transfers, "message": "Transfer history fetched successfully."}

    def get_bot_logs(self):
        try:
            with open("p2p_bot.log", "r") as f:
                lines = f.readlines()
                logs = "".join(lines[-500:])
            return {"logs": logs, "message": "Logs fetched successfully."}
        except FileNotFoundError:
            return {"logs": "", "message": "Log file not found."}
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return {"logs": "", "message": f"Error reading log file: {e}"}
        
    def _notify_user(self, order_id: str, msg: str):
        try:
            if getattr(self, "telegram_bot", None):
                _safe_fire_and_forget(
                    self.telegram_bot._safe_send(f"Order {order_id}: {msg}")
                )
        except Exception as e:
            logger.error(f"Notify user failed for {order_id}: {e}")

    def _send_bybit_chat_once(self, order_id: str, message: str, tag: str):
        """
        Send a Bybit chat message once per order per tag.
        tag examples: 'payment_started', 'payment_sent', 'issue'
        """
        if not self.redis_client:
            return

        redis_key = f"p2p_bot:chat_sent:{tag}"
        if self.redis_client.sismember(redis_key, order_id):
            return  # already sent

        resp = self.bybit_api.send_p2p_chat_message(order_id, message)
        if resp.get("ret_code") == 0:
            self.redis_client.sadd(redis_key, order_id)
            logger.info(f"📩 Sent Bybit chat ({tag}) for order {order_id}")
        else:
            logger.warning(f"⚠️ Failed to send chat for {order_id}: {resp}")

    def _execute_transfer(self, order_id, amount_naira, seller_bank_name, seller_account_no, seller_real_name):
     """Unified function to execute transfers via selected gateway."""

     lock_key = f"p2p_bot:lock:{order_id}"

     # Store order details FIRST (before any early returns)
     self._store_order_details(order_id, amount_naira, seller_bank_name, seller_account_no, seller_real_name)

     latest = self._latest_order_result(order_id)
     if not self._is_payable_status(latest):
        self._handle_transfer_failure(
            order_id,
            "Order no longer payable (appeal/paid/completed/cancelled/unknown).",
            log_level='warning'
        )
        return False

    
    # --- PREVENT DOUBLE PAYMENT ---
     if self.redis_client.sismember("p2p_bot:processed_orders", order_id) or self.redis_client.sismember("p2p_bot:pending_transfers", order_id):
        logger.warning(f"⛔ Order {order_id} already processed or being processed. Skipping to prevent double payment.")
        return False
     logger.info(f"Starting transfer process for order {order_id}")
     transfer_successful = False
     gateway_used = "Nomba"
     transfer_details = {}
     nomba_merchant_tx_ref = f"BYBIT_{order_id}"

     # Round amount to whole naira (remove kobo)
     amount_naira = self._round_amount(order_id, amount_naira)
     if amount_naira is False:  # rounding failed badly
      return False
     
     
     if not USE_NOMBA_FOR_TRANSFERS:
        logger.error(f"Nomba transfers disabled for order {order_id}")
        self._handle_transfer_failure(order_id,
                              "Nomba transfers disabled",
                              log_level='error')
        return False
        
     logger.info(f"Using Nomba API for order {order_id}")

     

     nomba_bank_code = get_bank_code(seller_bank_name, seller_account_no)
     if not nomba_bank_code:
         try:
            # Build a normalized lookup for fuzzy matching
            def _norm(s: str) -> str:
                if not s:
                    return ""
                s = s.lower()
                for w in ("bank", "mfb", "microfinance", "paymentservice", "psb"):
                    s = s.replace(w, "")
                return "".join(s.split())

            query_norm = _norm(seller_bank_name)
            
            norm_to_key = {_norm(k): k for k in BANK_CODES.keys() if k}

            # Fuzzy compare against normalized keys
            best = process.extractOne(
                query_norm,
                list(norm_to_key.keys()),
                scorer=fuzz.token_set_ratio
            )
            
            if best:
                best_norm_key, score = best[0], best[1]
                if score >= 80:
                    canonical_name = norm_to_key[best_norm_key]
                    nomba_bank_code = BANK_CODES.get(canonical_name)
                    logger.warning(
                        f"Fuzzy-corrected bank '{seller_bank_name}' -> "
                        f"'{canonical_name}' (code {nomba_bank_code}, score={score}) for {order_id}"
                    )
                    seller_bank_name = canonical_name  # update to canonical for logs/history
                else:
                    logger.error(
                        f"Fuzzy match too weak for '{seller_bank_name}' "
                        f"(best score={score}) on {order_id}"
                    )
            else:
                logger.error(f"No fuzzy match candidates for '{seller_bank_name}' on {order_id}")

         except Exception as e:
            logger.error(f"Fuzzy bank resolution error for '{seller_bank_name}' on {order_id}: {e}")

         if not nomba_bank_code:
                 self._handle_transfer_failure( order_id,f"No bank code for '{seller_bank_name}' ({seller_account_no}) after static+fuzzy lookup",log_level='error'
                 )
                 return False
     
     current_balance = self.nomba_api.get_wallet_balance()
     logger.info(f"Nomba balance for {order_id}: {current_balance:.2f} NGN")
     if current_balance == 0.0 or current_balance < amount_naira:
         logger.warning(f"Insufficient balance {current_balance:.2f} NGN for {amount_naira:.2f} NGN on {order_id}")
         self._handle_transfer_failure(order_id, f"Low balance for {order_id}", log_level='warning')
         return False
        
   

     if str(nomba_bank_code).startswith("9999") or any(
        w in seller_bank_name.lower() for w in ["opay", "palmpay", "psb", "smartcash", "momo", "9psb", ]
    ):
        logger.info(f"Skipping name check for wallet/PSB {seller_bank_name} on {order_id}")
        resolved_account_name = seller_real_name
     else:
          resolve_response = self.nomba_api.resolve_bank_account(normalize_account_number(seller_account_no), nomba_bank_code)
          logger.info(f"Nomba account lookup response for {order_id}: {resolve_response}")

      

          if not resolve_response or resolve_response.get('code') != '00':
            logger.error(f"Account lookup failed for {order_id}: {resolve_response.get('description')}")
            self._handle_transfer_failure(order_id, f"Lookup fail for {order_id}", log_level='error')
            return False
          resolved_account_name = resolve_response.get('data', {}).get('accountName') or seller_real_name

             
            #Name matching
     result = compare_names(seller_real_name, resolved_account_name) if resolved_account_name else (False, 0.0, 0, {"reason": "no resolution"})
     if not result or len(result) != 4:
            match, score, overlap, debug_info = False, 0.0, 0, {"reason": "compare_names failed"}
     else:
          match, score, overlap, debug_info = result
          logger.info(f"compare_names result for {order_id}: match={match}, score={score}")

          ok_to_continue = True if match else False

          if not match:
                score = fuzz.token_set_ratio(seller_real_name.lower(), resolved_account_name.lower())
                logger.info(f"Fuzzy score for {order_id}: {score}")
                ok_to_continue = score >= 0

                if score < 0:
                    logger.warning(f"Name mismatch for {order_id}: '{seller_real_name}' vs '{resolved_account_name}', score={score}")
                    self.redis_client.sadd("p2p_bot:review_orders", order_id)
                    self._notify_user(order_id, f"Name issue for {order_id}")
                    self._handle_transfer_failure(order_id, f"Name mismatch for {order_id}", log_level='warning')
                    return False
                else:
                    logger.info(f"Fuzzy match accepted (>=60) for {order_id}; proceeding.")
          if ok_to_continue:
           invalid_keywords = ["check dm", "read my terms", "whatsapp", "telegram", "agree", "inbox"]

    # normalize the account so spaces/dashes don’t break checks
           seller_account_no = normalize_account_number(seller_account_no)

          if any(word in f"{seller_bank_name} {seller_account_no}".lower() for word in invalid_keywords):
            logger.error(f"Invalid details keywords for {order_id}: {seller_bank_name} {seller_account_no}")
            self._handle_transfer_failure(order_id, f"Bad details for {order_id}", log_level='critical')
            return False

    # NUBAN is 10 digits; use <9 if you want to stay loose
          if len(seller_account_no) < 9:
           logger.error(f"Account length invalid for {order_id}: {seller_account_no} (len={len(seller_account_no)})")
           self._handle_transfer_failure(order_id, f"Bad details for {order_id}", log_level='critical')
           return False

        
       # --- MARK AS PENDING & STORE REFS ---
          self.redis_client.sadd("p2p_bot:pending_transfers", order_id)
          self.redis_client.hset("p2p_bot:pending_nomba_refs", order_id, nomba_merchant_tx_ref)
          self.redis_client.hset("p2p_bot:nomba_tx_ref_to_order", nomba_merchant_tx_ref, order_id)
          self.redis_client.hset("p2p_bot:pending_ts", order_id, str(time.time())) 
 
     try:
        # Initiate the transfer
         logger.info(f"Initiating transfer for {order_id}: {amount_naira} NGN to {seller_account_no}")
         transfer_response = self.nomba_api.initiate_fund_transfer(
            amount=amount_naira,
            account_number=normalize_account_number(seller_account_no),
            account_name=resolved_account_name or seller_real_name,
            bank_code=nomba_bank_code,
            merchant_tx_ref=nomba_merchant_tx_ref,
            narration=f"Order Payment - {order_id}"
        )
         logger.info(f"Nomba transfer response for {order_id}: {transfer_response}")
     except Exception as e:
        try:
            if self.redis_client:
                self.redis_client.srem("p2p_bot:pending_transfers", order_id)
                self.redis_client.hdel("p2p_bot:pending_nomba_refs", order_id)
                self.redis_client.hdel("p2p_bot:nomba_tx_ref_to_order", nomba_merchant_tx_ref)
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed during transfer initiation for {order_id}: {cleanup_error}")
        logger.error(f"Exception occurred during transfer initiation for {order_id}: {e}")
        self._handle_transfer_failure(order_id, f"Exception during transfer: {e}", log_level='error')
        return False
     transfer_status_code = transfer_response.get('code')
     transfer_description = transfer_response.get('description', 'No description')

     

    # Handle different response codes properly
     if transfer_status_code == '00':
            # Completed immediately
            logger.info(f"✅ Nomba transfer completed immediately for {order_id}")
            transfer_successful = True
            transfer_details= {"bybit_order_id": order_id,"nomba_transaction_reference": nomba_merchant_tx_ref,
                               "amount_naira": amount_naira, 
                               "recipient_bank": seller_bank_name,
                               "recipient_account": seller_account_no,
                               "gateway": "Nomba",
                               "nomba_status_code": transfer_status_code,
                               "timestamp_initiated": time.time(),
                               "status_details": transfer_description}

     elif transfer_status_code == '202':
        # Poll for status until success or timeout
        max_checks = 5
        check_delay = 5  # seconds
        for attempt in range(max_checks):
            time.sleep(check_delay)
            try:
                self.nomba_api._authenticate()  # Refresh token if needed
            except Exception as auth_error:
                logger.warning(f"Auth refresh failed on attempt {attempt + 1}: {auth_error}")
            try:
                status_response = self.nomba_api.check_transfer_status(nomba_merchant_tx_ref)
                logger.info(f"Status check attempt {attempt + 1} for {order_id}: {status_response}")
                status_code = status_response.get('code')
                status_flag = status_response.get('data', {}).get('status', '').upper()
                if status_code == "00" and status_flag == "SUCCESSFUL":
                    logger.info(f"✅ {order_id} now successful")
                    transfer_successful = True
                    transfer_details = {
                        "bybit_order_id": order_id,
                        "nomba_transaction_reference": nomba_merchant_tx_ref,
                        "amount_naira": amount_naira,
                        "recipient_bank": seller_bank_name,
                        "recipient_account": seller_account_no,
                        "gateway": "Nomba",
                        "nomba_status_code": status_code,
                        "timestamp_initiated": time.time(),
                        "status_details": "Success after recheck",
                    }
                    break
                elif status_code in ['403', 'Unauthorized']:
                    logger.warning(f"Token expired during status check for {order_id}, attempt {attempt + 1}")
                    continue
                elif 'fail' in status_response.get('description', '').lower():
                    logger.error(f"❌ Transfer failed for {order_id}: {status_response}")
                    break
                else:
                    logger.warning(f"⏳ {order_id} still processing, attempt {attempt + 1}: {status_response}")
            except AttributeError as e:
                logger.error(f"No check_transfer_status method available for {order_id}: {e}")
                self._handle_transfer_failure(
                    order_id,
                    f"Status check attribute error: {e}",
                    log_level='error'
                )
                break
            except Exception as e:
                logger.error(f"Error checking status for {order_id} on attempt {attempt + 1}: {e}")
                if attempt == max_checks - 1:
                    # Last attempt failed
                    self._handle_transfer_failure(
                        order_id,
                        f"Status check failed after {max_checks} attempts: {e}",
                        log_level='error'
                    )
                continue
        # If we couldn't confirm status but initiated successfully, assume success
        if not transfer_successful:
            logger.warning(f"⚠️ Could not confirm status for {order_id} but transfer was initiated. Assuming success.")
            transfer_successful = True  # Since money was debited
            transfer_details = {
                "bybit_order_id": order_id,
                "nomba_transaction_reference": nomba_merchant_tx_ref,
                "amount_naira": amount_naira,
                "recipient_bank": seller_bank_name,
                "recipient_account": seller_account_no,
                "gateway": "Nomba",
                "nomba_status_code": transfer_status_code,
                "timestamp_initiated": time.time(),
                "status_details": "Assumed success after unconfirmed status"
            }
            # Alert Telegram about unconfirmed transfer
            tg = getattr(self, "telegram_bot", None)
            if tg:
                try:
                    _safe_fire_and_forget(
                        tg.notify(
                            f"⚠️ UNCONFIRMED TRANSFER\n"
                            f"Order: {order_id}\n"
                            f"Amount: {amount_naira} NGN\n"
                            f"Bank: {seller_bank_name}\n"
                            f"Account: {seller_account_no}\n"
                            f"Status checks inconclusive. Transfer may have gone through.\n"
                            f"Monitor manually and confirm receipt."
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to send unconfirmed transfer alert to Telegram for {order_id}: {e}")
     else:
        # Unexpected status code (not 00, not 202)
        logger.error(f"❌ Unexpected Nomba response code {transfer_status_code} for {order_id}: {transfer_description}")
        self._handle_transfer_failure(
            order_id,
            f"Unexpected transfer response: {transfer_status_code} - {transfer_description}",
            log_level='error'
        )
        return False
            
         
        # ...after transfer_successful is set to True...
     if transfer_successful:
        logger.info(f"Fetching Bybit order details to confirm payment for {order_id}")
        details = self.bybit_api.get_order_details(order_id)
        if not details or details.get("ret_code") != 0:
          logger.error(f"Failed to fetch Bybit order details for {order_id}: {details.get('ret_msg', 'Unknown error')}")
          self._handle_transfer_failure(order_id, f"Failed to fetch Bybit order details for {order_id}", log_level='error')
          return False

        res = details.get("result")
        if not res:
            logger.error(f"No result data in Bybit order details for {order_id}")
            self._handle_transfer_failure(order_id, f"No result data in Bybit order details for {order_id}", log_level='error')
            return False

        bybit_payment_type, bybit_payment_id, pick_reason = self._pick_bybit_payment_term(res, seller_account_no)
        if not bybit_payment_type or not bybit_payment_id:
           logger.error(f"No valid payment term found for {order_id}. Marking as stuck.")
           self.redis_client.sadd("p2p_bot:stuck_orders", order_id)
           self.redis_client.hset("p2p_bot:stuck_order_details", order_id, json.dumps(details))
           self._handle_transfer_failure(order_id, f"No valid payment term found for {order_id}", log_level='error')
           return False

        logger.info(
          f"Confirming Bybit order {order_id} using paymentType={bybit_payment_type}, "
          f"paymentId={bybit_payment_id} (reason: {pick_reason})"
        )

        confirm_bybit_response = self.bybit_api.confirm_p2p_order_paid(
            order_id, bybit_payment_type, bybit_payment_id
        )

        ret_code = confirm_bybit_response.get('retCode', confirm_bybit_response.get('ret_code'))
        ret_msg = confirm_bybit_response.get('retMsg', confirm_bybit_response.get('ret_msg', 'Unknown error'))

         


        if ret_code == 0:
             logger.info(f"✅ Confirmed {order_id} as paid on Bybit")
             self._finalize_success(order_id, nomba_merchant_tx_ref, transfer_details)
             return True
            
    

     else:
         logger.error(f"❌ Bybit confirm failed for {order_id}: {ret_msg}")
         self.redis_client.sadd("p2p_bot:stuck_orders", order_id)
         self._handle_transfer_failure(order_id, f"Bybit confirm failed for {order_id}: {ret_msg}", log_level='error')
         return False
     if self.redis_client:
         self.redis_client.delete(lock_key)

     
    def _is_payable_status(self, order_result: dict) -> bool:
     """
    Only allow clearly payable orders.
    Reject appeal/cancel/paid/completed/unknown.
    Accept: 0=Unpaid, 10=Pending Payment (awaiting buyer).
    """
     if not order_result:
        return False

    # Numeric status (Bybit varies by endpoint; keep defensive)
     st_num = order_result.get("status") or order_result.get("orderStatus")
    # Textual hints (different fields exist across payloads)
     st_txt = (order_result.get("statusStr")
              or order_result.get("statusDesc")
              or order_result.get("orderStatusDesc")
              or "").upper()
    # Dedicated appeal flags if present
     in_appeal = bool(order_result.get("inAppeal")
                     or order_result.get("isAppeal")
                     or order_result.get("appealStatus"))

    # Hard reject if appeal is flagged anywhere
     if in_appeal:
        return False

    # Common numeric map: 0=Unpaid, 1=Paid, 2=Completed, 3=Cancelled, 10=Pending Payment
     reject_numeric = {1, 2, 3}
     acceptable_numeric = {0, 10}

     if isinstance(st_num, int):
        if st_num in reject_numeric:
            return False
        return st_num in acceptable_numeric

    # Fallback to text
     if any(w in st_txt for w in ("APPEAL", "CANCEL", "COMPLETED", "PAID", "HOLD")):
        return False
     if any(w in st_txt for w in ("UNPAID", "PENDING", "WAIT", "AWAIT")):
        return True

    # Unknown => be safe, do not pay
     return False


    def _latest_order_result(self, order_id: str) -> dict:
        """
        Safely fetch latest order details; returns {} on any problem.
        """
        try:
            resp = self.bybit_api.get_order_details(order_id)
            if not resp or resp.get("ret_code", resp.get("retCode")) != 0:
                return {}
            return resp.get("result") or {}
        except Exception:
            return {}


     

    def _pick_bybit_payment_term(self, order_result, seller_account_no):
     payment_terms = order_result.get("paymentTermList", [])
     seller_norm = normalize_account_number(seller_account_no)
     for term in payment_terms:
        acc = normalize_account_number(term.get("accountNo", ""))
        if acc and acc == seller_norm:
            return term.get("paymentType"), term.get("id"), "Matched by normalized account number"
    # Fallback as you had
     if payment_terms:
        term = payment_terms[0]
        return term.get("paymentType"), term.get("id"), "Defaulted to first payment term"
     return None, None, "No payment term found"

    
    def retry_failed_order(self, order_id, bank, account, name):
        logger.info(f"Retrying failed order {order_id} with updated details.")
        # Load original amount from Redis
        amount_naira = self._rget(f'p2p_bot:order_details:{order_id}', 'amount')
        try:
            amount_naira = float(amount_naira)
        except Exception:
            amount_naira = None
        self._execute_transfer(order_id, amount_naira,
                              seller_bank_name=bank,
                              seller_account_no=account,
                              seller_real_name=name)
    def get_success_rate(self):
        if not self.redis_client:
            return {"success_rate": None, "message": "Redis not connected."}
        total_attempted = (
            self.redis_client.scard("p2p_bot:processed_orders") +
            self.redis_client.scard("p2p_bot:stuck_orders") +
            self.redis_client.scard("p2p_bot:insufficient_funds_orders")
        )
        successful = self.redis_client.scard("p2p_bot:processed_orders")
        if total_attempted == 0:
            rate = 0.0
        else:
            rate = successful / total_attempted * 100
        return {
            "success_rate": round(rate, 2),
            "successful": successful,
            "total_attempted": total_attempted
        }
    
    def _round_amount(self, order_id, amount_naira):
     try:
        amt_dec = Decimal(str(amount_naira))
        amt_whole = amt_dec.quantize(Decimal('1'), rounding=ROUND_DOWN)  # e.g. 145000.34 -> 145000
        if amt_whole != amt_dec:
            logger.info(f"Rounding down amount for {order_id} from {amt_dec} to {amt_whole} (remove kobo).")
        amount_naira = int(amt_whole)  # keep as int for transfer API
     except Exception as e:
        logger.warning(f"Failed to round amount for {order_id}: {e}. Falling back to int floor.")
        try:
            amount_naira = int(float(amount_naira))
        except Exception:
            # if totally bad, abort
            self._handle_transfer_failure(order_id, "Invalid amount; cannot round.", log_level='error')
            return False
     return amount_naira
        






                

                        
                    


    def process_p2p_orders(self):
        logger.info("--- P2P Bot Cycle Started ---")
        if not self.redis_client:
            logger.critical("Redis client not connected. Skipping bot cycle.")
            return

        lock_key = "p2p_bot:lock"
        lock_acquired = self.redis_client.set(lock_key, "locked", ex=60, nx=True)
        if not lock_acquired:
            logger.warning("Bot is already processing. Skipping this cycle to prevent concurrency issues.")
            return

        try:
            self.redis_client.set('p2p_bot:last_cycle_time', time.time())
            
            # Call get_p2p_orders with fixed parameters for side="Buy" and status="Pending"
            pending_orders = self.bybit_api.get_pending_orders()

            if not pending_orders:
                logger.info("No pending Bybit orders found.")
                return

            logger.info(f"Found {len(pending_orders)} pending Bybit orders.")

            if len(pending_orders) > MAX_ORDERS_PER_CYCLE:
                logger.info(f"Limiting processing to {MAX_ORDERS_PER_CYCLE} orders out of {len(pending_orders)} found.")
                pending_orders = pending_orders[:MAX_ORDERS_PER_CYCLE]

            logger.info(f"Found {len(pending_orders)} pending Bybit orders for processing.")

            for order in pending_orders:
                order_id = order.get('id') or order.get('orderId')
                if not order_id:
                    logger.warning(f"Skipping malformed order without ID: {order}")
                    continue

                amount_usdt = float(order.get('quantity') or order.get('notifyTokenQuantity') or 0)
                amount_naira = float(order.get('amount') or 0)

                

                
                logger.debug(f"Evaluating order {order_id} for payment processing.")

                # Check if order is already processed or marked for skipping
                if self.redis_client.sismember("p2p_bot:processed_orders", order_id):
                    logger.info(f"⏭ Skipping Bybit order {order_id}: Already processed in a previous cycle.")
                    continue
                if self.redis_client.sismember("p2p_bot:stuck_orders", order_id):
                    logger.warning(f"⏭ Skipping Bybit order {order_id}: Marked as stuck. Manual intervention required.")
                    continue
                if self.redis_client.sismember("p2p_bot:insufficient_funds_orders", order_id):
                    logger.warning(f"⏭ Skipping Bybit order {order_id}: Marked as insufficient funds. Manual intervention required.")
                    continue
                if self.redis_client.sismember("p2p_bot:cancelled_by_user_orders", order_id):
                    logger.info(f"⏭ Skipping Bybit order {order_id}: Manually cancelled by user.")
                    continue
                
    

                # Fetch detailed order info from Bybit (P2P API)
                order_info_response = self.bybit_api.get_order_details(order_id,)
                if not order_info_response or order_info_response.get('ret_code') != 0:
                    self._handle_transfer_failure(order_id, f"Failed to get detailed info for order {order_id}: {order_info_response.get('ret_msg', 'Unknown error')}")
                    continue
                
                order_details = order_info_response['result']

                # NEW: don’t process if order is not clearly payable (e.g., appealed / paid / completed / cancelled)
                if not self._is_payable_status(order_details):
                 self._handle_transfer_failure(
                  order_id,
                          "Order not in a payable state (appeal/paid/completed/cancelled/unknown).",
                   log_level='warning'
                )
                 continue


                if not order_details:
                    self._handle_transfer_failure(order_id, f"No order details found for order {order_id}. Skipping.")
                    continue

                # FIX: Update amount_naira from detailed order info
                amount_naira = float(order_details.get('amount') or 0)

                # Store amount early so if extraction fails, Telegram gets the order amount
                if self.redis_client:
                    self.redis_client.hset(f'p2p_bot:order_details:{order_id}', 'amount', str(amount_naira))

                # Extract seller details robustly using your fallback function
                try:
                    seller_bank_name, seller_account_no, seller_real_name = self._extract_payment_details(order_details, order_id)
                except Exception as extract_error:
                    logger.error(f"Extraction exception for {order_id}: {extract_error}")
                    self._handle_transfer_failure(
                        order_id,
                        f"Payment details extraction crashed: {extract_error}",
                        log_level='error'
                    )
                    continue

                if not all([seller_bank_name, seller_account_no, seller_real_name]):
                    # Extraction returned None - notify admin to fix manually
                    logger.warning(f"Extraction incomplete for {order_id}. Requesting admin intervention.")
                    self.redis_client.sadd("p2p_bot:extraction_failed_orders", order_id)
                    self.redis_client.hset(f'p2p_bot:order_details:{order_id}', mapping={
                        'bank': seller_bank_name or 'EXTRACTION_FAILED',
                        'account': seller_account_no or 'EXTRACTION_FAILED',
                        'name': seller_real_name or 'EXTRACTION_FAILED',
                        'amount': str(amount_naira),
                        'timestamp': str(time.time())
                    })
                    # Send Telegram notification asking admin to fix
                    tg = getattr(self, "telegram_bot", None)
                    if tg:
                        try:
                            error_details = {
                                'order_id': order_id,
                                'reason': 'Standard extraction + AI fallback both failed',
                                'amount': amount_naira,
                                'seller_bank_name': seller_bank_name or 'UNKNOWN',
                                'seller_account_no': seller_account_no or 'UNKNOWN',
                                'seller_real_name': seller_real_name or 'UNKNOWN'
                            }
                            _safe_fire_and_forget(
                                tg.send_stuck_order_notification(
                                    order_id,
                                    "Payment details extraction failed (standard + AI). Please provide correct details.",
                                    error_details
                                )
                            )
                            logger.info(f"Sent manual review request to Telegram for {order_id}")
                        except Exception as tg_error:
                            logger.error(f"Failed to send extraction failure alert to Telegram for {order_id}: {tg_error}")
                    continue  # skip if details not valid


                self._store_order_details(order_id, amount_naira, seller_bank_name, seller_account_no, seller_real_name)
                if self.use_approval_mode and not self.redis_client.sismember('p2p_bot:approved_orders', order_id):
                    already_notified = self.redis_client.sismember('p2p_bot:approval_notified_orders', order_id)
                    if not already_notified and getattr(self, "telegram_bot", None):
                        try:
                            payload = {
                                "order_id": order_id,
                                "amount_naira": amount_naira,
                                "seller_bank_name": seller_bank_name,
                                "seller_account_no": seller_account_no,
                                "seller_real_name": seller_real_name
                            }
                            _safe_fire_and_forget(
                                self.telegram_bot.send_order_approval_request(order_id, payload)
                            )
                            self.redis_client.sadd('p2p_bot:approval_notified_orders', order_id)
                            logger.info(f"Sent approval request for order {order_id} to Telegram for {order_id}")
                        except Exception as e:
                            logger.error(f"Failed to send approval request for order {order_id}: {e}")
                            self._handle_transfer_failure(order_id, f"Failed to send approval request: {e}", log_level='error')
                        continue
                    

                # Execute transfer using the unified function
                self._execute_transfer(order_id, amount_naira, seller_bank_name, seller_account_no, seller_real_name)

                self.redis_client.delete(lock_key)
                logger.info(f"Processed order {order_id} successfully.")

            logger.info("--- P2P Bot Cycle Completed ---")

        except Exception as e:
            logger.critical(f"An unhandled error occurred during bot cycle: {e}", exc_info=True)
        finally:
            if self.redis_client and self.redis_client.get(lock_key) == "locked":
                self.redis_client.delete(lock_key)
                logger.debug("Released processing lock.")

    def check_pending_nomba_transfers(self):
        pending_orders = list(self.redis_client.smembers("p2p_bot:pending_transfers"))
        for order_id in pending_orders:
            nomba_merchant_tx_ref = self.redis_client.hget("p2p_bot:pending_nomba_refs", order_id)
            if not nomba_merchant_tx_ref:
                logger.error(f"No merchant_tx_ref found for pending order {order_id}. Skipping status check.")
                continue
            status_resp = self.nomba_api._send_request(
                "GET",
                f"/v1/transfers/bank/{nomba_merchant_tx_ref}",
                requires_auth=True
            )
            code = status_resp.get("code")
            desc = status_resp.get("description", "")
            if code == "00":
                logger.info(f"Nomba transfer for order {order_id} now successful. Marking as processed.")
                self.redis_client.srem("p2p_bot:pending_transfers", order_id)
                self.redis_client.hdel("p2p_bot:pending_nomba_refs", order_id)
                self.redis_client.sadd("p2p_bot:processed_orders", order_id)
                confirm_resp = self.bybit_api.confirm_p2p_order_paid(order_id)
                if confirm_resp.get("retCode") == 0:
                    logger.info(f"Confirmed order {order_id} as paid on Bybit after pending transfer.")
                else:
                    logger.error(f"Failed to confirm order {order_id} as paid on Bybit after pending transfer: {confirm_resp.get('retMsg')}")
            elif code not in ("202", "00"):
                logger.error(f"Nomba transfer for order {order_id} failed or unknown status ({code}: {desc}). Marking as stuck.")
                self.redis_client.srem("p2p_bot:pending_transfers", order_id)
                self.redis_client.hdel("p2p_bot:pending_nomba_refs", order_id)
                self.redis_client.sadd("p2p_bot:stuck_orders", order_id)
            else:
                logger.info(f"Nomba transfer for order {order_id} still processing ({code}: {desc}). Will check again next cycle.")

    def _finalize_success(self, order_id, nomba_merchant_tx_ref, transfer_details):
      try:
        self._send_bybit_chat_once(
            order_id,
            "✅ Paid please confirm and release, positive review will be highly appreciated. Thank you.",
            tag="payment_sent"
        )

        if self.redis_client:
            # clear pending
            self.redis_client.srem("p2p_bot:pending_transfers", order_id)
            self.redis_client.hdel("p2p_bot:pending_nomba_refs", order_id)
            if nomba_merchant_tx_ref:
                self.redis_client.hdel("p2p_bot:nomba_tx_ref_to_order", nomba_merchant_tx_ref)

            # mark processed + store history
            self.redis_client.sadd("p2p_bot:processed_orders", order_id)
            if transfer_details:
                self.redis_client.lpush("p2p_bot:transfers", json.dumps(transfer_details))

        # Telegram success
        if getattr(self, "telegram_bot", None):
            _safe_fire_and_forget(self.telegram_bot.send_success_notification(order_id, transfer_details))
      except Exception as e:
        logger.error(f"Finalize success cleanup failed for {order_id}: {e}")
    


# Define a placeholder fallback_lookup function
# Initialize P2PBotService with required dependencies
bybit_api = BybitP2PClient(BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_BASE_URL)
nomba_api = NombaAPI(NOMBA_CLIENT_ID, NOMBA_CLIENT_SECRET, NOMBA_ACCOUNT_ID, NOMBA_BASE_URL, NOMBA_SENDER_NAME)
p2p_bot_service = P2PBotService(bybit_api, nomba_api, redis_client)
# Start APScheduler immediately at boot so the bot runs even with no incoming HTTP traffic

scheduler_initialized = False

@app.before_request
def initialize_scheduler_once():
    global scheduler_initialized
    if not scheduler_initialized:
        if not hasattr(app, 'apscheduler') or not getattr(app, 'apscheduler', None):
            scheduler = APScheduler()
            scheduler.init_app(app)
            app.apscheduler = scheduler   # 👈 keep a strong reference here

            logger.info("APScheduler initialized with Flask app.")

            if not scheduler.get_job('process_p2p_orders_job'):
                scheduler.add_job(
                    id='process_p2p_orders_job',
                    func=p2p_bot_service.process_p2p_orders,
                    trigger='interval',
                    seconds=POLLING_INTERVAL_SECONDS
                )
                logger.info(f"Scheduler job 'process_p2p_orders_job' added with interval {POLLING_INTERVAL_SECONDS} seconds.")

            if redis_client:
                if not scheduler.running:
                    scheduler.start()
                    logger.info("APScheduler started automatically upon backend initialization.")
            else:
                logger.critical("Redis not connected, APScheduler will not start automatically.")
        scheduler_initialized = True
        # ---- START SCHEDULER AT BOOT (place this right after initialize_scheduler_once) ----
        with app.app_context():
         try:
             initialize_scheduler_once()  # ensures app.apscheduler exists and job is added
             if hasattr(app, "apscheduler") and not app.apscheduler.running and redis_client:
              app.apscheduler.start()
              logger.info("APScheduler started at boot via app context.")
         except Exception as e:
          logger.error(f"Failed to start scheduler at boot: {e}")
# ---- END SCHEDULER AT BOOT ----





# --- Helper for Bank Codes (Improved Normalization and Data) ---
# This dictionary should ideally be populated dynamically from Nomba/Paystack APIs
# For now, it's a cleaned-up static list.
BANK_CODES = {
    "uba": "033",
    "nombank": "090645",
    "5tt": "090832",
    "9japay": "090629",
    "9psb": "120001",
    "ab": "090270",
    "agmortgage": "418",
    "amju": "090180",
    "arm": "090816",
    "asosavings&loans": "090001",
    "abbey": "070010",
    "abucoop": "820",
    "access": "044",
    "accessyellow": "100052",
    "accion": "090134",
    "adamawa": "070030",
    "addosser": "090160",
    "aella": "090614",
    "airtelsmartcashpsb": "120004",
    "alternative": "100029",
    "ampersand": "090529",
    "assetmatrix": "090287",
    "auchi": "090264",
    "auchipoly": "090817",
    "avuenegbe": "090478",
    "blooms": "090743",
    "bank78": "110072",
    "bankofagriculture": "090367",
    "baobab": "090136",
    "bestar": "090615",
    "betastacktechnologies": "110074",
    "boost": "090819",
    "bowen": "50931",
    "bowman": "090804",
    "branchinternationalfinancialservices": "050006",
    "budinfrastructure": "983",
    "build": "090613",
    "capricondigital": "956",
    "carbon": "100026",
    "cashconnect": "748",
    "charis": "090815",
    "chukwunenye": "090490",
    "citibanknigerialimited": "023",
    "conteglobalinfotechlimited": "100032",
    "corestep": "766",
    "creditville": "090611",
    "crust": "090560",
    "dsc": "090821",
    "davodani": "50159",
    "davodanimicrofinance": "090391",
    "diamond": "063",
    "dillon": "090828",
    "doje": "090404",
    "dot": "090470",
    "esettlementltd.": "999999",
    "eastman": "090707",
    "ecobanknigeria": "050",
    "ekondo": "090097",
    "enrich": "090539",
    "enterprise": "084",
    "fsdhmerchant": "400001",
    "fairmoney": "090551",
    "fidelity": "070",
    "finatrust": "090111",
    "first": "011",
    "firstbank": "011",
    "firstcitymonument": "214",
    "fcmb": "214",
    "firstmarinatrustlimited": "050022",
    "firstmonniewallet": "309",
    "flexi": "090835",
    "flutterwave": "622",
    "gtbank": "058",
    "globus": "000027",
    "goldman": "090574",
    "goodnews": "090495",
    "greenacres": "090599",
    "grooming": "090195",
    "habaripay": "110059",
    "hackman": "090147",
    "halacredit": "090291",
    "heritage": "030",
    "hopepaymentservicebank": "120002",
    "ibile": "090118",
    "ikoyiosun": "090536",
    "jaiz": "301",
    "jubileelifemortgage": "090003",
    "kenechukwu": "090602",
    "keystone": "082",
    "kolomoni": "899",
    "kongapay": "100025",
    "kredi": "090380",
    "kuda": "090267",
    "livingtrustmortgage": "070007",
    "lomabank": "090620",
    "lapo": "090177",
    "leadcity": "397",
    "letshego": "090420",
    "lotus": "000029",
    "mainstreet": "090171",
    "momopaymentservicebank": "120003",
    "kuda": "090267",
    "livingtrustmortgage": "070007",
    "lomabank": "090620",
    "lapo": "090177",
    "leadcity": "397",
    "letshego": "090420",
    "lotus": "000029",
    "mainstreet": "090171",
    "momopaymentservicebank": "120003",
    "moneytronics": "090692",
    "moniepoint": "090405",
    "moremonee": "090685",
    "npf": "070001",
    "netappstechnology": "950",
    "nirsal": "090194",
    "noun": "090822",
    "oau": "090345",
    "palmpay": "100033",
    "parallexmf": "000030",
    "parkway-readycash": "100003",
    "payattitudeonline": "329",
    "paycom(opay)": "305",
    "paystacktitan": "100039",
    "projectsmicrofinance": "090503",
    "prospacapital": "50739",
    "providus": "101",
    "randalpha": "090496",
    "renmoney": "090198",
    "royalexchange": "090138",
    "rubies": "090175",
    "safehaven": "090286",
    "sciartfinance": "050024",
    "shalom": "090502",
    "smartcashpsb": "942",
    "sparkle": "090325",
    "spectrummfb": "090436",
    "stanbicibtc": "039",
    "standardcharteredbanknigeria": "068",
    "stellas": "667",
    "sterling": "232",
    "summit": "080003",
    "suntrust": "100",
    "taj": "000026",
    "tatumbank": "000042",
    "tellerone": "090788",
    "titantrust": "000025",
    "toprate": "090801",
    "unn": "090251",
    "uda": "672",
    "ukpor": "090820",
    "umuoji": "090814",
    "unical": "090193",
    "union": "032",
    "unitedbankforafrica": "033",
    "unity": "215",
    "vfdmicrofinancebanklimited": "566",
    "valefinance": "050020",
    "victory": "090813",
    "wema": "035",
    "whitecrustfinance": "050035",
    "xpressmts": "148",
    "xpresspayments": "738",
    "xpresswallet": "391",
    "yellodigitalservices": "964",
    "zwallet": "792",
    "zenith": "057",
    "zikora": "090504",
    "etranzact": "306"
}

def get_bank_code(bank_name: str, account_number: Optional[str] = None) -> Optional[str]:
    """
    Resolves a bank name to its corresponding bank code using a multi-layered approach:
    1. Rule-based fixes for common wallet mislabels.
    2. Fuzzy matching against a static dictionary.
    3. GPT-based intelligent fallback for unlisted or misspelled banks.
    """
    if not bank_name:
        return None

    # --- Step 1: Rule-Based Correction for Wallet Mislabels ---
    if account_number:
        acct_str = str(account_number)
        if ("momo" in bank_name.lower() or "psb" in bank_name.lower()):
            if acct_str.startswith(("90", "70")):
                logger.info(f"Corrected '{bank_name}' to Opay based on account number prefix.")
                return BANK_CODES.get("paycom(opay)")
            elif acct_str.startswith(("80", "81")):
                logger.info(f"Corrected '{bank_name}' to Palmpay based on account number prefix.")
                return BANK_CODES.get("palmpay")
    
    # --- Step 2: Normalize and Search Static Dictionary ---
    normalized_input = bank_name.lower()
    for word in ["bank", "mfb", "microfinance", "paymentservice", "psb"]:
        normalized_input = normalized_input.replace(word, "")
    normalized_input = "".join(normalized_input.split())

    for key, code in BANK_CODES.items():
        clean_key = key.lower()
        for word in ["bank", "mfb", "microfinance", "paymentservice", "psb"]:
            clean_key = clean_key.replace(word, "")
        clean_key = "".join(clean_key.split())

        if normalized_input in clean_key or clean_key in normalized_input:
            logger.info(f"Found static match for '{bank_name}' -> '{key}' ({code})")
            return code
            
    # --- Step 3: GPT-based Fallback ---
    logger.warning(f"Static lookup failed for '{bank_name}'. Falling back to GPT.")
    
    try:
        # Assuming your gpt_bank_name_to_code function returns a tuple (name, code)
        corrected = gpt_bank_name_to_code(bank_name, BANK_CODES) 
        if corrected:
            corrected_name, corrected_code = corrected
            logger.info(f"GPT successfully resolved '{bank_name}' to '{corrected_name}' with code {corrected_code}.")
            return corrected_code
        else:
            logger.warning(f"GPT could not resolve bank name '{bank_name}'. No match found.")
            return None
    except Exception as e:
        logger.error(f"GPT fallback for bank name failed: {e}")
        return None
import difflib

def resolve_bank_code(user_input, bank_code_dict, threshold=0.8):
    """
    Resolves a user's inputted bank name to the correct bank code using fuzzy matching.
    
    Args:
        user_input (str): The bank name entered by the user.
        bank_code_dict (dict): Dictionary of normalized bank names to bank codes.
        threshold (float): Confidence threshold (0.0 to 1.0). Default is 0.8.
        
    Returns:
        str: The matched bank code.
        
    Raises:
        ValueError: If no suitable match is found above the threshold.
    """
    def normalize(name):
        if not name:
            return ""
        name = name.lower().replace("bank", "").replace("mfb", "").replace("microfinance", "")
        name = "".join(name.split())
        return name

    user_norm = normalize(user_input)
    best_match = None
    best_score = 0.0

    for bank_name, code in bank_code_dict.items():
        bank_norm = normalize(bank_name)
        score = difflib.SequenceMatcher(None, user_norm, bank_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = code

    if best_score >= threshold:
        return best_match
    else:
        raise ValueError(f"No suitable bank match found for '{user_input}' (best score: {best_score:.2f})")

# Example usage:
# BANK_CODES = {"first bank": "011", "gtbank": "058", ...}
# code = resolve_bank_code("Fst Bank", BANK_CODES)

# --- Flask Routes (No changes to existing routes) ---

        
@app.route('/webhook/nomba/transfer', methods=['POST'])
def nomba_transfer_webhook():
    data = request.json
    logger.info(f"Received Nomba webhook: {data}")

    merchant_tx_ref = data.get("merchantTxRef") or data.get("meta", {}).get("merchantTxRef")
    status = data.get("status") or data.get("data", {}).get("status")
    order_id = None

    # If you store mapping order_id <-> merchant_tx_ref, retrieve order_id here
    if merchant_tx_ref:
        order_id = redis_client.hget("p2p_bot:nomba_tx_ref_to_order", merchant_tx_ref)
        if not order_id:
            logger.warning(f"Received webhook for unknown merchantTxRef: {merchant_tx_ref}")
        else:
            logger.info(f"Webhook for order {order_id}, merchantTxRef {merchant_tx_ref}, status {status}")

        if status and status.upper() == "SUCCESSFUL":
            # Mark as processed
            redis_client.sadd("p2p_bot:processed_orders", order_id)
            redis_client.srem("p2p_bot:pending_transfers", order_id)
            redis_client.hdel("p2p_bot:pending_nomba_refs", order_id)
            # Confirm to Bybit
            resp = p2p_bot_service.bybit_api.confirm_p2p_order_paid(order_id)
            logger.info(f"Bybit confirm after webhook for {order_id}: {resp}")
        elif status and status.upper() in ("FAILED", "REVERSED"):
            redis_client.srem("p2p_bot:pending_transfers", order_id)
            redis_client.hdel("p2p_bot:pending_nomba_refs", order_id)
            redis_client.sadd("p2p_bot:stuck_orders", order_id)
            logger.error(f"Transfer for order {order_id} failed or reversed by Nomba webhook.")
    else:
        logger.warning("No merchantTxRef in webhook payload.")

        if status and status.upper() == "SUCCESSFUL":
         details = p2p_bot_service.bybit_api.get_order_details(order_id)
         res = details.get("result", {}) if details else {}
        acct = redis_client.hget(f"p2p_bot:order_details:{order_id}", "account")
        ptype, pid, _ = p2p_bot_service._pick_bybit_payment_term(res, acct)
        confirm = p2p_bot_service.bybit_api.confirm_p2p_order_paid(order_id, ptype, pid)
        rc = confirm.get('retCode', confirm.get('ret_code'))
        if rc == 0:
          td = {"bybit_order_id": order_id, "nomba_transaction_reference": merchant_tx_ref, "status_details": "Webhook success"}
          p2p_bot_service._finalize_success(order_id, merchant_tx_ref, td)   # <— ADD THIS
        else:
            p2p_bot_service._handle_transfer_failure(order_id, f"Bybit confirm failed via webhook: {confirm}", 'error')


    return jsonify({"status": "ok"})

# app.py
from flask import jsonify
# START
@app.route('/control/start', methods=['POST'])
@app.route('/api/control/start', methods=['POST'])
def start_bot():
    initialize_scheduler_once()
    if hasattr(app, 'apscheduler'):
        job = app.apscheduler.get_job('process_p2p_orders_job')
        if not job:
            app.apscheduler.add_job(
                id='process_p2p_orders_job',
                func=p2p_bot_service.process_p2p_orders,
                trigger='interval',
                seconds=POLLING_INTERVAL_SECONDS
            )
        if not app.apscheduler.running:
            app.apscheduler.start()
        return jsonify({
            "status": "success",
            "running": app.apscheduler.running,
            "job_exists": app.apscheduler.get_job('process_p2p_orders_job') is not None
        }), 200

# STOP (do NOT shutdown the scheduler; just remove/pause the job)
@app.route('/control/stop', methods=['POST'])
@app.route('/api/control/stop', methods=['POST'])
def control_stop():
    sched = getattr(app, "apscheduler", None)
    if not sched:
        return jsonify({"ok": True, "running": False, "note": "no scheduler"}), 200
    if sched.get_job('process_p2p_orders_job'):
        sched.remove_job('process_p2p_orders_job')  # or .pause()
    return jsonify({
        "ok": True,
        "running": bool(sched.running),
        "job_exists": sched.get_job('process_p2p_orders_job') is not None
    }), 200

# STATUS
@app.route('/control/status', methods=['GET'])
@app.route('/api/control/status', methods=['GET'])
def control_status():
    sched = getattr(app, "apscheduler", None)
    job = sched.get_job('process_p2p_orders_job') if sched else None
    last_cycle = None
    try:
        if redis_client:
            v = redis_client.get('p2p_bot:last_cycle_time')
            last_cycle = float(v) if v else None
    except Exception:
        pass
    return jsonify({
        "running": bool(sched and sched.running),
        "job_exists": job is not None,
        "next_run": str(job.next_run_time) if job else None,
        "last_cycle": last_cycle,
        "use_approval_mode": getattr(p2p_bot_service, "use_approval_mode", None),
        "use_nomba_for_transfers": USE_NOMBA_FOR_TRANSFERS,
    }), 200

# SUCCESS RATE
@app.route('/control/success-rate', methods=['GET'])
@app.route('/api/control/success-rate', methods=['GET'])
def control_success_rate():
    data = p2p_bot_service.get_success_rate()
    return jsonify({"ok": True, **data}), 200
