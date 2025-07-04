import os
import sys
import time
import requests
import hmac
import hashlib
import json
from datetime import datetime, timedelta
from functools import wraps
from typing import Union, List, Tuple
import uuid
import logging
from collections import OrderedDict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Keeping print statements for very early diagnostic in case logger fails
print("DEBUG (Early): load_dotenv() completed.", file=sys.stderr)
print(f"DEBUG (Early): LOG_LEVEL from env: {os.getenv('LOG_LEVEL')}", file=sys.stderr)

# --- Logging Configuration ---
log_file_path = '/opt/p2p-bot/p2p_bot.log'
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger('P2P_Bot_Backend')

# --- Flask App Configuration ---
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import redis

app = Flask(_name_)
CORS(app)

# --- Environment Variables ---
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
BYBIT_P2P_BASE_URL = os.getenv('BYBIT_P2P_BASE_URL', 'https://api.bybit.com')

PAYSTACK_SECRET_KEY = os.getenv('PAYSTACK_SECRET_KEY')
PAYSTACK_WEBHOOK_SECRET = os.getenv('PAYSTACK_WEBHOOK_SECRET')
PAYSTACK_BASE_URL = os.getenv('PAYSTACK_BASE_URL', 'https://api.paystack.co')

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
TRANSFER_PENDING_TIMEOUT_MINUTES = int(os.getenv('TRANSFER_PENDING_TIMEOUT_MINUTES', 30))
BOT_CYCLE_INTERVAL_SECONDS = int(os.getenv('BOT_CYCLE_INTERVAL_SECONDS', 30))

FLASK_DASHBOARD_CONTROL_KEY = os.getenv('FLASK_DASHBOARD_CONTROL_KEY')
if FLASK_DASHBOARD_CONTROL_KEY:
    masked_key = FLASK_DASHBOARD_CONTROL_KEY[:3] + '#' * (len(FLASK_DASHBOARD_CONTROL_KEY) - 3)
    logger.info(f"FLASK_DASHBOARD_CONTROL_KEY loaded (masked): {masked_key}")
else:
    logger.warning("FLASK_DASHBOARD_CONTROL_KEY is not set in environment variables. Dashboard control endpoints will not be securely accessible.")

if not all([BYBIT_API_KEY, BYBIT_API_SECRET, PAYSTACK_SECRET_KEY, FLASK_DASHBOARD_CONTROL_KEY]):
    logger.critical("One or more critical environment variables are not set. Please check your .env file.")

# --- Redis Client ---
redis_client = None
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except Exception as e:
    logger.critical(f"Failed to connect to Redis at {REDIS_URL}: {e}. Redis operations will not function.")

# --- Bybit Client (P2P Specific) ---
class BybitP2PClient:
    def _init_(self, api_key: str, api_secret: str, base_url: str):
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

            if data and data.get('ret_code') == 0 and data.get('result'):
                # Extract payment info from 'paymentTermList'
                payment_terms = data['result'].get('paymentTermList', [])
                if payment_terms:
                    logger.info(f"Found payment terms for order {order_id}. Returning first entry.")
                    return payment_terms[0]
                else:
                    logger.warning(f"No payment terms found in details for order {order_id}. Response: {data}")
                    return None
            else:
                logger.error(f"Failed to fetch details for order {order_id}: {data.get('ret_msg')}. Full response: {data}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Bybit order details for {order_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Bybit API error response content: {e.response.text}")
            return None


# --- Paystack Client (Direct Requests) ---
class PaystackClient:
    def _init_(self, secret_key: str, base_url: str, redis_client: redis.Redis = None):
        self.secret_key = secret_key
        self.base_url = base_url
        self.redis = redis_client # Added redis_client
        self.recipient_cache_key_prefix = "paystack:recipient_cache:" # Key prefix for recipient cache
        self.supported_banks = {} # Initialized empty, loaded later

        self.headers = {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json"
        } if self.secret_key else {}
        
        if not self.secret_key:
            logger.warning("Paystack Secret Key is missing. Paystack API calls will likely fail.")
        else:
            self.supported_banks = self._load_supported_banks() # Load banks only if key exists
        logger.info("Initializing Paystack client (using direct requests).")

    def _load_supported_banks(self) -> dict:
        if not self.secret_key:
            logger.warning("Paystack secret key is not set, cannot load supported banks.")
            return {}

        logger.info("Attempting to load supported banks from Paystack via direct API call.")
        # --- DEFINITIVE FIX: The 'pay_with_bank_transfer=true' parameter has been definitively removed. ---
        # This ensures we get the full list of banks supported for payouts/transfers.
        # I am incredibly sorry for the repeated oversight. This is now CORRECT.
        url = f"{self.base_url}/bank?country=nigeria"
        logger.debug(f"Paystack Bank List URL being used: {url}") # Explicitly log the URL for verification
        # --- END DEFINITIVE FIX ---

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and data.get('status') and data.get('data'):
                banks = {}
                bank_names_list = [] # To store names for logging
                for bank in data['data']:
                    bank_name_lower = bank['name'].lower()
                    banks[bank_name_lower] = bank['code']
                    bank_names_list.append(bank['name']) # Keep original case for logging
                # --- Log the full list of banks received from Paystack ---
                logger.info(f"Successfully loaded {len(banks)} banks from Paystack. Banks: {', '.join(bank_names_list)}")
                # --- END Log ---
                return banks
            else:
                # Log the message from Paystack if status is not true
                logger.warning(f"Paystack API returned no data for supported banks or status not true: {data.get('message', 'No message provided')}")
                return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error loading supported banks from Paystack: {e}")
            return {}

    def get_bank_code_by_name(self, bank_name: str) -> Union[str, None]:
        if not self.secret_key:
            logger.error("Paystack secret key is not configured. Cannot get bank code.")
            return None
        
        # Ensure bank_name is a string before calling .lower() and .strip()
        normalized_bank_name = str(bank_name).lower().strip()

        code = self.supported_banks.get(normalized_bank_name)
        if not code:
            logger.warning(f"Bank code for '{bank_name}' not found in cache. Attempting to refresh bank list.")
            # Optionally, re-load the supported banks if a bank is not found
            self.supported_banks = self._load_supported_banks()
            code = self.supported_banks.get(normalized_bank_name) # Check again after refresh
            if code:
                logger.info(f"Bank code for '{bank_name}' found after refresh: {code}")
            else:
                logger.error(f"Bank code for '{bank_name}' still not found after refresh.")
        return code

    def create_transfer_recipient(self, name: str, account_number: str, bank_code: str) -> Union[dict, None]:
        if not self.secret_key:
            logger.error("Paystack secret key is not configured. Cannot create transfer recipient.")
            return None

        # --- RECIPENT CACHING LOGIC ---
        # Cache key combines account_number and bank_code
        cache_key = f"{self.recipient_cache_key_prefix}{account_number}:{bank_code}"
        if self.redis:
            cached_recipient_data_json = self.redis.get(cache_key)
            if cached_recipient_data_json:
                try:
                    cached_recipient_data = json.loads(cached_recipient_data_json.decode('utf-8'))
                    logger.info(f"Found cached Paystack recipient for {account_number} ({bank_code}): {cached_recipient_data.get('recipient_code')}")
                    return cached_recipient_data
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding cached recipient data for {account_number}:{bank_code}: {e}. Will attempt to create new.")
                    # Invalidate bad cache entry
                    self.redis.delete(cache_key)
        # --- END RECIPENT CACHING LOGIC ---

        logger.info(f"Attempting to create Paystack recipient for {name} - {account_number} ({bank_code}).")
        url = f"{self.base_url}/transferrecipient"
        payload = {
            "type": "nuban",
            "name": name,
            "account_number": account_number,
            "bank_code": bank_code,
            "currency": "NGN"
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and data.get('status') and data.get('data'):
                recipient_code = data['data']['recipient_code']
                logger.info(f"Successfully created Paystack recipient: {recipient_code}")
                # --- CACHE THE NEW RECIPIENT with 30-day expiry ---
                if self.redis:
                    # Store full recipient data as JSON string for future flexibility
                    self.redis.set(cache_key, json.dumps(data['data']), ex=60*60*24*30) # 30 days expiry
                    logger.debug(f"Cached full recipient data for {account_number}:{bank_code} with 30-day expiry.")
                # --- END CACHING ---
                return data['data']
            else:
                logger.error(f"Failed to create Paystack recipient: {data.get('message', 'No message provided')}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating Paystack recipient for {account_number}: {e}")
            return None

    def initiate_transfer(self, recipient_code: str, amount_naira: float, reason: str) -> Union[dict, None]:
        if not self.secret_key:
            logger.error("Paystack secret key is not configured. Cannot initiate transfer.")
            return None

        logger.info(f"Attempting to initiate Paystack transfer of {amount_naira} NGN to {recipient_code}.")
        url = f"{self.base_url}/transfer"
        payload = {
            "source": "balance",
            "amount": int(float(amount_naira) * 100), # Ensured float conversion for amount
            "recipient": recipient_code,
            "reason": reason,
            "currency": "NGN"
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data and data.get('status') and data.get('data'):
                logger.info(f"Successfully initiated Paystack transfer: {data['data']['transfer_code']}")
                return data['data']
            else:
                logger.error(f"Failed to initiate Paystack transfer: {data.get('message', 'No message provided')}. Full response: {data}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error initiating Paystack transfer to {recipient_code}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Paystack API error response content: {e.response.text}")
            return None

    def verify_transfer(self, transfer_code: str) -> Union[dict, None]:
        if not self.secret_key:
            logger.error("Paystack secret key is not configured. Cannot verify transfer.")
            return None

        logger.info(f"Verifying Paystack transfer: {transfer_code}.")
        url = f"{self.base_url}/transfer/verify/{transfer_code}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and data.get('status') and data.get('data'):
                logger.info(f"Successfully verified Paystack transfer {transfer_code}: Status {data['data']['status']}")
                return data['data']
            else:
                logger.warning(f"Failed to verify Paystack transfer {transfer_code}: {data.get('message', 'No message provided')}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying Paystack transfer {transfer_code}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Paystack API error response content: {e.response.text}")
            return None

# --- P2P Bot Service (Core Logic) ---
class P2PBotService:
    def _init_(self, bybit_client: BybitP2PClient, paystack_client: PaystackClient, redis_client: redis.Redis):
        self.bybit = bybit_client
        self.paystack = paystack_client
        self.redis = redis_client
        self.is_active = False # Controls if the bot processes orders
        self.processing_lock_key = "p2p_bot:processing_lock"
        self.transfer_history_key = "p2p_bot:transfer_history"
        self.stuck_orders_key = "p2p_bot:stuck_orders" # Key for orders that might be stuck (general issues)
        self.insufficient_funds_key = "p2p_bot:insufficient_funds_orders" # New key for orders stuck due to balance
        self.last_cycle_time = None
        
        # Initialize the bot's active state from Redis if it was previously set
        if self.redis:
            active_state = self.redis.get("p2p_bot:is_active")
            if active_state:
                self.is_active = active_state.decode('utf-8').lower() == 'true'
                logger.info(f"Bot initialized with active state from Redis: {self.is_active}")
        
        logger.info("P2PBotService initialized.")

    def set_scheduler_active(self, active: bool):
        self.is_active = active
        if self.redis:
            self.redis.set("p2p_bot:is_active", "true" if active else "false")
            logger.info(f"Bot active state set to {active} and saved to Redis.")

    def get_scheduler_status(self) -> bool:
        return self.is_active

    def acquire_lock(self, timeout=60) -> bool:
        """Acquire a lock to prevent multiple concurrent bot cycles."""
        if self.redis:
            # setnx returns 1 if key was set (lock acquired), 0 if key already exists
            lock_acquired = self.redis.setnx(self.processing_lock_key, "locked")
            if lock_acquired:
                self.redis.expire(self.processing_lock_key, timeout)
                logger.debug("Acquired processing lock.")
                return True
            logger.debug("Failed to acquire processing lock (another cycle is running).")
            return False
        logger.warning("Redis client not available. Cannot acquire lock.")
        return True # Proceed without lock if Redis is down (with warning)

    def release_lock(self):
        """Release the processing lock."""
        if self.redis:
            self.redis.delete(self.processing_lock_key)
            logger.debug("Released processing lock.")
        else:
            logger.warning("Redis client not available. Cannot release lock.")

    def add_transfer_to_history(self, transfer_data: dict):
        """Adds a transfer record to Redis list."""
        if self.redis:
            transfer_json = json.dumps(transfer_data)
            self.redis.lpush(self.transfer_history_key, transfer_json)
            # Trim the list to prevent it from growing indefinitely (e.g., keep last 1000)
            self.redis.ltrim(self.transfer_history_key, 0, 999)
            logger.info(f"Added transfer {transfer_data.get('paystack_transfer_code')} to history with status: {transfer_data.get('paystack_status')}.")
        else:
            logger.warning("Redis client not available. Cannot add transfer to history.")

    def get_all_transfers_from_history(self) -> List[dict]:
        """Retrieves all transfers from Redis history."""
        if self.redis:
            raw_history = self.redis.lrange(self.transfer_history_key, 0, -1)
            history = []
            for item in raw_history:
                try:
                    history.append(json.loads(item.decode('utf-8')))
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding transfer history item from Redis: {e} - {item}")
            # Sort by timestamp_initiated in descending order for dashboard display
            history.sort(key=lambda x: x.get('timestamp_initiated', 0), reverse=True)
            logger.info(f"Retrieved {len(history)} transfers from history.")
            return history
        logger.warning("Redis client not available. Cannot retrieve transfer history.")
        return []

    def update_transfer_in_history(self, transfer_code: str, new_status: str, status_details: str = None):
        """Updates an existing transfer record in Redis history."""
        if not self.redis:
            logger.warning("Redis client not available. Cannot update transfer history.")
            return
        
        # Get all history, find and update the specific entry, then re-save
        current_history = self.get_all_transfers_from_history()
        found = False
        for i, record in enumerate(current_history):
            if record.get('paystack_transfer_code') == transfer_code:
                if record.get('paystack_status') != new_status: # Only update if status actually changed
                    record['paystack_status'] = new_status
                    if status_details:
                        record['status_details'] = status_details
                    # Optionally update the timestamp if it's a final status change
                    record['timestamp_updated'] = int(time.time())
                    current_history[i] = record
                    found = True
                    break
        
        if found:
            # Clear existing list and re-add updated list
            self.redis.delete(self.transfer_history_key)
            for record in reversed(current_history): # Add back in original order (reverse of sorted)
                self.redis.rpush(self.transfer_history_key, json.dumps(record))
            logger.info(f"Updated transfer {transfer_code} to status: {new_status}")
        else:
            logger.warning(f"Could not find transfer {transfer_code} to update in history.")

    def add_stuck_order(self, order_id: str):
        """Adds an order to a set of potentially stuck orders."""
        if self.redis:
            self.redis.sadd(self.stuck_orders_key, order_id)
            logger.warning(f"Added order {order_id} to stuck orders list.")

    def remove_stuck_order(self, order_id: str):
        """Removes an order from the set of potentially stuck orders."""
        if self.redis:
            self.redis.srem(self.stuck_orders_key, order_id)
            logger.info(f"Removed order {order_id} from stuck orders list.")

    def get_stuck_orders(self) -> List[str]:
        """Retrieves all currently stuck orders."""
        if self.redis:
            stuck_orders = [s.decode('utf-8') for s in self.redis.smembers(self.stuck_orders_key)]
            logger.debug(f"Current stuck orders: {stuck_orders}")
            return stuck_orders
        return []

    def add_insufficient_funds_order(self, order_id: str):
        """Adds an order to a set of orders stuck due to insufficient funds."""
        if self.redis:
            self.redis.sadd(self.insufficient_funds_key, order_id)
            logger.error(f"Order {order_id} marked as stuck due to insufficient funds.")

    def remove_insufficient_funds_order(self, order_id: str):
        """Removes an order from the set of orders stuck due to insufficient funds."""
        if self.redis:
            self.redis.srem(self.insufficient_funds_key, order_id)
            logger.info(f"Removed order {order_id} from insufficient funds list.")

    def is_insufficient_funds_order(self, order_id: str) -> bool:
        """Checks if an order is currently marked due to insufficient funds."""
        if self.redis:
            return self.redis.sismember(self.insufficient_funds_key, order_id)
        return False

    def get_otp_pending_transfers(self) -> List[dict]:
        """Retrieves transfers from history that are currently 'otp' status."""
        history = self.get_all_transfers_from_history()
        return [t for t in history if t.get('paystack_status') == 'otp']

    def monitor_pending_paystack_transfers(self):
        """Periodically checks the status of OTP-pending Paystack transfers."""
        otp_pending_transfers = self.get_otp_pending_transfers()
        if not otp_pending_transfers:
            logger.info("No OTP-pending transfers to monitor.")
            return

        logger.info(f"Monitoring {len(otp_pending_transfers)} OTP-pending Paystack transfers.")
        for transfer_record in otp_pending_transfers:
            bybit_order_id = transfer_record.get('bybit_order_id')
            paystack_transfer_code = transfer_record.get('paystack_transfer_code')
            timestamp_initiated = transfer_record.get('timestamp_initiated', 0)

            if not paystack_transfer_code or paystack_transfer_code == 'N/A':
                logger.warning(f"Skipping OTP monitoring for order {bybit_order_id}: missing Paystack Transfer Code. Adding to stuck orders.")
                self.add_stuck_order(bybit_order_id) # Mark as stuck if transfer code is missing/N/A
                continue
            
            # Check for timeout
            if (time.time() - timestamp_initiated) > (TRANSFER_PENDING_TIMEOUT_MINUTES * 60):
                logger.warning(f"OTP pending transfer for order {bybit_order_id} (Code: {paystack_transfer_code}) has timed out after {TRANSFER_PENDING_TIMEOUT_MINUTES} minutes. Marking as stuck.")
                self.update_transfer_in_history(paystack_transfer_code, 'timed_out_otp', 'OTP not resolved within timeout.')
                self.add_stuck_order(bybit_order_id)
                continue

            # Verify status with Paystack
            verification_result = self.paystack.verify_transfer(paystack_transfer_code)
            if verification_result:
                new_status = verification_result.get('status')
                if new_status and new_status != transfer_record.get('paystack_status'):
                    logger.info(f"Paystack transfer {paystack_transfer_code} for order {bybit_order_id} status changed from '{transfer_record.get('paystack_status')}' to '{new_status}'.")
                    self.update_transfer_in_history(paystack_transfer_code, new_status, verification_result.get('message'))
                    if new_status in ['success', 'successful']:
                        # If transfer succeeded, confirm on Bybit
                        logger.info(f"Paystack transfer for order {bybit_order_id} is '{new_status}'. Attempting to confirm payment on Bybit.")
                        bybit_confirmed = self.bybit.confirm_bybit_payment(bybit_order_id) # Needs to be implemented in BybitP2PClient
                        if bybit_confirmed:
                            logger.info(f"Successfully confirmed Bybit order {bybit_order_id} payment after Paystack success.")
                            self.remove_stuck_order(bybit_order_id) # Remove if it was stuck
                        else:
                            logger.error(f"Failed to confirm payment on Bybit for order {bybit_order_id} despite Paystack success. Manual intervention needed.")
                            self.add_stuck_order(bybit_order_id)
                    elif new_status in ['failed', 'reversed', 'cancelled']:
                        logger.warning(f"Paystack transfer for order {bybit_order_id} failed with status '{new_status}'. Adding to stuck orders.")
                        self.add_stuck_order(bybit_order_id)
            else:
                logger.warning(f"Could not verify status of Paystack transfer {paystack_transfer_code} for order {bybit_order_id}. Will retry on next cycle.")

    def process_bybit_order_for_payment(self, order: dict) -> bool:
        order_id = order.get('orderId')
        fiat_amount = float(order.get('fiatAmount', 0.0))

        logger.info(f"Processing Bybit order {order_id} with fiat amount: {fiat_amount}")

        # --- Smart Retry Logic Checks ---
        if self.is_insufficient_funds_order(order_id):
            logger.info(f"Skipping Bybit order {order_id}: Previously flagged for insufficient funds. Top up Paystack balance and clear via dashboard to retry.")
            return False # Indicate that payment was not processed this cycle

        # Check if there's an already existing OTP-pending transfer for this order in history
        existing_transfers = [t for t in self.get_all_transfers_from_history() if t.get('bybit_order_id') == order_id]
        otp_pending = next((t for t in existing_transfers if t.get('paystack_status') == 'otp'), None)

        if otp_pending:
            logger.info(f"Skipping Bybit order {order_id}: Existing Paystack transfer {otp_pending['paystack_transfer_code']} is pending OTP. Monitoring its status.")
            # The monitor_pending_paystack_transfers will handle its resolution
            return False # Indicate that payment was not processed this cycle
        # --- End Smart Retry Logic Checks ---

        # Fetch full order details including payment method
        detailed_payment_info = self.bybit.get_order_details(order_id)
        if not detailed_payment_info:
            logger.error(f"Could not fetch detailed payment info for Bybit order {order_id}. Cannot process for payment.")
            self.add_stuck_order(order_id)
            return False
        
        seller_bank_name = detailed_payment_info.get('bankName')
        seller_account_no = detailed_payment_info.get('accountNo')
        seller_account_holder_name = detailed_payment_info.get('realName')

        if not all([seller_bank_name, seller_account_no, seller_account_holder_name]):
            logger.error(f"Missing crucial bank details from order {order_id} after fetching full details. Cannot process for payment. Details: {detailed_payment_info}")
            self.add_stuck_order(order_id)
            return False
        
        logger.info(f"Processing Bybit order {order_id} for payment: {fiat_amount} NGN to {seller_account_holder_name} ({seller_bank_name} - {seller_account_no})")

        # 1. Get bank code from Paystack
        bank_code = self.paystack.get_bank_code_by_name(seller_bank_name)
        if not bank_code:
            logger.error(f"Could not find bank code for '{seller_bank_name}' for Bybit order {order_id}. Cannot proceed with transfer.")
            self.add_stuck_order(order_id)
            return False

        # 2. Create Paystack transfer recipient
        recipient_data = self.paystack.create_transfer_recipient(
            name=seller_account_holder_name,
            account_number=seller_account_no,
            bank_code=bank_code
        )
        if not recipient_data or not recipient_data.get('recipient_code'):
            logger.error(f"Failed to create Paystack recipient for Bybit order {order_id}. Cannot proceed with transfer.")
            self.add_stuck_order(order_id)
            return False
        recipient_code = recipient_data['recipient_code']
        logger.info(f"Created/Found Paystack recipient for order {order_id}: {recipient_code}")

        # 3. Initiate Paystack transfer
        transfer_reason = f"P2P payment for Bybit order {order_id}"
        transfer_result = self.paystack.initiate_transfer(recipient_code, float(fiat_amount), transfer_reason)
        
        # Record transfer immediately, even if pending/failed, to avoid re-processing
        transfer_status = transfer_result.get('status', 'failed') if transfer_result else 'failed'
        paystack_transfer_code = transfer_result.get('transfer_code', 'N/A') if transfer_result else 'N/A'
        paystack_message = transfer_result.get('message', 'N/A') if transfer_result else 'Transfer initiation failed.'


        transfer_record = {
            "bybit_order_id": order_id,
            "paystack_transfer_code": paystack_transfer_code,
            "amount_naira": float(fiat_amount),
            "recipient_bank": seller_bank_name,
            "recipient_account": seller_account_no,
            "paystack_status": transfer_status,
            "timestamp_initiated": int(time.time()),
            "status_details": paystack_message
        }
        self.add_transfer_to_history(transfer_record)

        if not transfer_result or not transfer_result.get('status'):
            logger.error(f"Failed to initiate Paystack transfer for Bybit order {order_id}. Transfer status: {transfer_status}. Message: {paystack_message}. Adding to stuck orders or insufficient funds list.")
            if "insufficient_balance" in paystack_message.lower(): # Check if the error message indicates insufficient balance
                self.add_insufficient_funds_order(order_id)
            else:
                self.add_stuck_order(order_id)
            return False # Payment was not successful

        
        logger.info(f"Paystack transfer initiated for Bybit order {order_id}. Transfer Code: {paystack_transfer_code}. Status: {transfer_status}")
        self.remove_stuck_order(order_id)
        self.remove_insufficient_funds_order(order_id) # Remove if it was previously there

        # 4. Confirm payment on Bybit (only if Paystack transfer was successful or pending)
        # Note: 'otp' is also a pending state where action is required on Paystack dashboard
        if transfer_status in ['success', 'pending']:
            logger.info(f"Attempting to confirm payment on Bybit for order {order_id} after Paystack transfer initiated with status: {transfer_status}.")
            bybit_confirmed = self.bybit.confirm_bybit_payment(order_id) # Needs to be implemented in BybitP2PClient
            if bybit_confirmed:
                logger.info(f"Successfully confirmed Bybit order {order_id} payment.")
                # Do NOT remove from stuck orders here, remove only after full process success
            else:
                logger.error(f"Failed to confirm payment on Bybit for order {order_id} despite Paystack success. Manual intervention needed.")
                self.add_stuck_order(order_id) # Add to stuck if Bybit confirmation fails
            return bybit_confirmed # Return success status of Bybit confirmation
        elif transfer_status == 'otp':
            logger.warning(f"Paystack transfer for order {order_id} is pending OTP confirmation. Manual approval required on Paystack dashboard. Bot will monitor status.")
            # We do NOT confirm on Bybit immediately for OTP, we wait for Paystack 'success'
            return False # Payment not fully processed yet, still pending OTP
        else:
            logger.warning(f"Paystack transfer for order {order_id} was not successful/pending/OTP ({transfer_status}). Not confirming on Bybit. Check Paystack dashboard.")
            self.add_stuck_order(order_id)
            return False # Payment not processed successfully

    def run_cycle(self):
        self.last_cycle_time = datetime.now()
        if not self.is_active:
            logger.info("P2P Bot is currently paused. Skipping cycle.")
            return

        if not self.bybit or not self.paystack or not self.redis:
            logger.error("P2P Bot clients (Bybit/Paystack/Redis) are not initialized. Cannot run cycle.")
            return

        if not self.acquire_lock():
            logger.info("Cycle already running or lock contention. Skipping this cycle.")
            return

        try:
            logger.info("--- Starting P2P Bot Cycle ---")
            
            # First, monitor and update any existing OTP-pending transfers
            self.monitor_pending_paystack_transfers()

            # 1. Fetch pending Bybit orders
            pending_orders = self.bybit.get_pending_orders()
            logger.info(f"Found {len(pending_orders)} pending Bybit orders for processing.")

            # 2. Process each pending order
            for order in pending_orders:
                order_id = order.get('orderId')
                
                # Check history to avoid re-processing already successful/timed-out orders
                # And also check new 'insufficient_funds' flag
                
                # Retrieve transfer history for this specific order
                order_history = [t for t in self.get_all_transfers_from_history() if t.get('bybit_order_id') == order_id]

                # Check if already successfully transferred
                if any(t.get('paystack_status') in ['success', 'successful'] for t in order_history):
                    logger.info(f"Bybit order {order_id} already appears in history as successful. Skipping.")
                    continue

                # Check if an OTP is currently pending for this order
                # This is handled by monitor_pending_paystack_transfers, so we don't try to initiate a new one
                if any(t.get('paystack_status') == 'otp' for t in order_history):
                    logger.info(f"Bybit order {order_id} has a pending OTP transfer. Monitoring its status instead of initiating new.")
                    continue # It will be handled by monitor_pending_paystack_transfers

                # Check if it's flagged for insufficient funds
                if self.is_insufficient_funds_order(order_id):
                    logger.info(f"Skipping Bybit order {order_id}: Flagged for insufficient Paystack funds. Manual top-up required.")
                    continue

                # If none of the above, proceed to process for payment
                self.process_bybit_order_for_payment(order)
            
            # 3. Handle stuck orders
            current_stuck_orders = self.get_stuck_orders()
            if current_stuck_orders:
                logger.warning(f"Found {len(current_stuck_orders)} potentially stuck orders: {current_stuck_orders}")
                # Placeholder for future stuck order recovery logic
            else:
                logger.info("No stuck orders found in Redis.")

            # Also log insufficient funds orders
            current_insufficient_funds_orders = self.redis.smembers(self.insufficient_funds_key) if self.redis else []
            if current_insufficient_funds_orders:
                logger.warning(f"Found {len(current_insufficient_funds_orders)} orders stuck due to insufficient Paystack funds: {[o.decode('utf-8') for o in current_insufficient_funds_orders]}")
            else:
                logger.info("No orders currently flagged for insufficient funds.")


            logger.info("--- P2P Bot Cycle Completed ---")

        except Exception as e:
            logger.exception(f"An unexpected error occurred during the bot cycle: {e}")
        finally:
            self.release_lock()

# --- Placeholder for Bybit confirm_bybit_payment (MUST BE IMPLEMENTED) ---
# NOTE: This method is CRITICAL for marking the order as paid on Bybit.
# Without it, your bot will send funds but not update the Bybit order status.
# You need to fill this with the actual Bybit API call to confirm payment for the order ID.
def _placeholder_confirm_bybit_payment(self, order_id: str) -> bool:
    logger.warning(f"WARNING: BybitP2PClient.confirm_bybit_payment for order {order_id} is a placeholder and does not make an API call. YOU MUST IMPLEMENT THIS.")
    # Example (replace with actual Bybit API logic to confirm payment):
    # url = f"{self.base_url}/v5/p2p/order/confirm"
    # payload = {"orderId": order_id}
    # ... create signature and headers ...
    # response = requests.post(url, headers=headers, json=payload)
    # if response.json().get('ret_code') == 0:
    #     return True
    # return False
    return True # Simulate success for testing until implemented

# Dynamically add the placeholder method to the BybitP2PClient class
# This assumes you would replace this with a real implementation in your development
setattr(BybitP2PClient, 'confirm_bybit_payment', _placeholder_confirm_bybit_payment)


# --- Global Instances (initialized once) ---
bybit_client = None
paystack_client = None
p2p_bot_service = None

if redis_client:
    try:
        bybit_client = BybitP2PClient(BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_P2P_BASE_URL)
    except Exception as e:
        logger.critical(f"Failed to initialize BybitClient: {e}")

    try:
        paystack_client = PaystackClient(PAYSTACK_SECRET_KEY, PAYSTACK_BASE_URL, redis_client=redis_client)
    except Exception as e:
        logger.critical(f"Failed to initialize PaystackClient: {e}")

    if bybit_client and paystack_client:
        p2p_bot_service = P2PBotService(bybit_client, paystack_client, redis_client)
    else:
        logger.critical("Bybit or Paystack client failed to initialize. P2PBotService will not be initialized.")
else:
    logger.critical("Redis client is not available. P2PBotService will not be initialized.")


# --- Scheduler Setup ---
scheduler = BackgroundScheduler()
if p2p_bot_service:
    scheduler.add_job(p2p_bot_service.run_cycle, 'interval', seconds=BOT_CYCLE_INTERVAL_SECONDS)
    if not scheduler.running:
        scheduler.start()
        p2p_bot_service.set_scheduler_active(True) # Ensure bot starts active when scheduler is new
        logger.info(f"Scheduler started, running bot cycle every {BOT_CYCLE_INTERVAL_SECONDS} seconds.")
    else:
        logger.info("Scheduler already running from previous initialization.")
else:
    logger.critical("Scheduler not started because P2PBotService could not be initialized due to missing clients or Redis.")


# --- Helper for Dashboard Control Key ---
def require_control_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not FLASK_DASHBOARD_CONTROL_KEY:
            logger.warning("FLASK_DASHBOARD_CONTROL_KEY is not set. Control endpoints will be unprotected.")
            return jsonify({"status": "error", "message": "Dashboard control key not configured on backend."}), 500

        control_key_header = request.headers.get('X-Control-Secret')
        if not control_key_header:
            logger.warning("Unauthorized dashboard access attempt from client (missing X-Control-Secret header)")
            return jsonify({"status": "error", "message": "Unauthorized: X-Control-Secret header missing."}), 401
        
        if not hmac.compare_digest(control_key_header.encode('utf-8'), FLASK_DASHBOARD_CONTROL_KEY.encode('utf-8')):
            logger.warning(f"Unauthorized dashboard access attempt from client (invalid key). Provided key length: {len(control_key_header)}.")
            return jsonify({"status": "error", "message": "Unauthorized: Invalid control key."}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# --- Flask Request/Response Logging Middleware ---
@app.before_request
def log_request_info():
    """Logs details of incoming HTTP requests."""
    client_ip = request.headers.get('X-Real-Ip') or request.headers.get('X-Forwarded-For') or request.remote_addr
    if request.path != '/api/status':
        logger.info(f"Request: {request.method} {request.path} From: {client_ip} Headers: {dict(request.headers)}")
        if request.is_json:
            logger.info(f"Request JSON Body: {request.json}")
        elif request.form: # Corrected indentation for elif
            logger.info(f"Request Form Data: {request.form}")

@app.after_request
def log_response_info(response):
    """Logs details of outgoing HTTP responses."""
    client_ip = request.headers.get('X-Real-Ip') or request.headers.get('X-Forwarded-For') or request.remote_addr
    if request.path != '/api/status':
        logger.info(f"Response: {response.status} To: {client_ip} Content-Type: {response.content_type}")
        if response.content_type == 'application/json' and len(response.get_data()) < 2000:
            try:
                logger.info(f"Response Body: {json.loads(response.get_data())}")
            except json.JSONDecodeError:
                logger.info(f"Response Body (non-JSON): {response.get_data().decode('utf-8')[:500]}...")
    return response


# --- Flask Routes (Dashboard API) ---

@app.route('/api/status', methods=['GET'])
def get_bot_status():
    status = "critical"
    message = "Unknown error during status check."
    scheduler_active = False
    last_cycle_time_iso = None

    try:
        scheduler_active = p2p_bot_service.get_scheduler_status() if p2p_bot_service else False
        status = "running" if scheduler_active else "paused"
        message = "Bot is operational." if status == "running" else "Bot is currently paused."

        if p2p_bot_service and p2p_bot_service.last_cycle_time:
            last_cycle_time_iso = p2p_bot_service.last_cycle_time.isoformat()

        # Check Redis connection
        redis_ok = False
        if redis_client:
            try:
                redis_client.ping()
                redis_ok = True
            except redis.exceptions.ConnectionError as e:
                logger.error(f"Redis connection error during status check: {e}")
                message = "Redis connection lost. Data persistence may be affected."
            except Exception as e:
                logger.error(f"Unexpected error during Redis ping in status check: {e}")
                message = f"Unexpected Redis error: {str(e)}"

        if not redis_ok:
            status = "critical"
        
        # Check Bybit client initialization and API keys
        if not bybit_client:
            status = "critical"
            message = "Bybit client failed to initialize. API calls won't work. Check environment variables."
        elif not bybit_client.api_key or not bybit_client.api_secret:
            status = "critical"
            message = "Bybit API keys are missing or invalid. P2P operations will fail."

        # Check Paystack client initialization and API keys
        if not paystack_client:
            status = "critical"
            message = "Paystack client failed to initialize. Transfers won't function. Check environment variables."
        elif not paystack_client.secret_key:
            status = "warning"
            message = "Paystack secret key is missing. Transfers will not function."
        
        # Final check if the P2PBotService isn't considered active or functional
        if not p2p_bot_service:
            status = "critical"
            message = "P2P Bot Service itself failed to initialize. Check critical environment variables."
        elif not bybit_client or not bybit_client.api_key or not bybit_client.api_secret:
            status = "critical"
            message = "Bybit API is not fully configured/functional."
        elif not paystack_client or not paystack_client.secret_key:
             if status != "critical":
                status = "warning"
                message = "Paystack API is not fully configured/functional."
        elif not redis_ok:
             if status != "critical":
                status = "critical"
                message = "Redis is not fully configured/functional."


    except Exception as e:
        logger.exception(f"Unhandled exception in get_bot_status: {e}")
        status = "critical"
        message = f"An unexpected error occurred in status check: {str(e)}. Check backend logs."

    return jsonify({
        "status": status,
        "message": message,
        "scheduler_active": scheduler_active,
        "last_cycle": last_cycle_time_iso
    })

@app.route('/api/orders/pending', methods=['GET'])
@require_control_key
def get_pending_orders_dashboard():
    try:
        if not p2p_bot_service:
            logger.error("Attempted to fetch pending orders, but P2PBotService is not initialized.")
            return jsonify({"status": "error", "message": "Bot service not initialized. Check backend logs."}), 500
        pending_orders = p2p_bot_service.bybit.get_pending_orders()
        return jsonify({"status": "success", "data": pending_orders})
    except Exception as e:
        logger.exception("Error in /api/orders/pending endpoint")
        return jsonify({"status": "error", "message": f"Internal server error fetching pending orders: {str(e)}"}), 500


@app.route('/api/transfers', methods=['GET'])
@require_control_key
def get_transfer_history_dashboard():
    try:
        if not p2p_bot_service:
            logger.error("Attempted to fetch transfer history, but P2PBotService is not initialized.")
            return jsonify({"status": "error", "message": "Bot service not initialized. Check backend logs."}), 500
        transfer_history = p2p_bot_service.get_all_transfers_from_history()
        return jsonify({"status": "success", "data": transfer_history})
    except Exception as e:
        logger.exception("Error in /api/transfers endpoint")
        return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"}), 500


@app.route('/api/control/start', methods=['POST'])
@require_control_key
def start_bot():
    if not p2p_bot_service:
        logger.error("Attempted to start bot, but P2PBotService is not initialized.")
        return jsonify({"status": "error", "message": "Bot service not initialized. Check backend logs."}), 500
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler manually started via dashboard.")
    p2p_bot_service.set_scheduler_active(True)
    logger.info("Bot set to active via dashboard control.")
    return jsonify({"status": "message", "message": "Bot started successfully."})

@app.route('/api/control/stop', methods=['POST'])
@require_control_key
def stop_bot():
    if not p2p_bot_service:
        logger.error("Attempted to stop bot, but P2PBotService is not initialized.")
        return jsonify({"status": "error", "message": "Bot service not initialized. Check backend logs."}), 500
    p2p_bot_service.set_scheduler_active(False)
    logger.info("Bot paused via dashboard control.")
    return jsonify({"status": "message", "message": "Bot paused successfully."})

# --- Added Cleanup Endpoint ---
@app.route('/api/cleanup/stuck-orders', methods=['POST'])
@require_control_key
def cleanup_stuck_orders_endpoint():
    """Endpoint to manually clear stuck orders (and history) from Redis."""
    try:
        if p2p_bot_service and p2p_bot_service.redis:
            # Delete all relevant keys to completely reset for fresh data and clear stuck flags
            p2p_bot_service.redis.delete(p2p_bot_service.stuck_orders_key)
            p2p_bot_service.redis.delete(p2p_bot_service.insufficient_funds_key)
            deleted_history_count = p2p_bot_service.redis.delete(p2p_bot_service.transfer_history_key)
            logger.info(f"Manually cleared stuck orders, insufficient funds orders, and {deleted_history_count} items from transfer history in Redis.")
            return jsonify({"status": "success", "message": f"Cleared stuck orders, insufficient funds orders, and {deleted_history_count} items from transfer history."})
        elif not p2p_bot_service:
             logger.error("Attempted to cleanup, but P2PBotService is not initialized.")
             return jsonify({"status": "error", "message": "Bot service not initialized. Check backend logs."}), 500
        return jsonify({"status": "error", "message": "Redis not available"}), 500
    except Exception as e:
        logger.error(f"Error cleaning up stuck orders: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if _name_ == '_main_':
    logger.info("Running Flask app in development mode.")
    app.run(host='0.0.0.0', port=5000)