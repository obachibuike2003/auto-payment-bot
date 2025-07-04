# P2P Crypto Trading Bot Backend (Bybit & Paystack Integration)

## 🎯 Project Goal
This project implements an automated Python backend service designed to streamline Peer-to-Peer (P2P) cryptocurrency trading operations on Bybit. It automates the process of making fiat payments for pending Bybit P2P orders by integrating with the Paystack payment gateway, manages transaction states using Redis, and provides a Flask-based API for real-time monitoring and control.

## ✨ Why this Project is Important
This project demonstrates the ability to build a complex, real-world automation system by integrating multiple external APIs (Bybit, Paystack), managing state with a database (Redis), and providing a secure control interface (Flask API). It showcases practical skills in:
- **API Integration:** Securely interacting with third-party financial APIs.
- **Asynchronous Processing:** Using a background scheduler for continuous operation.
- **State Management:** Implementing robust state tracking and error handling for financial transactions.
- **System Design:** Structuring a multi-component application (clients, service, API, scheduler).
- **Security:** Handling sensitive API keys and securing dashboard access.

## 🚀 Key Features & Concepts Demonstrated
- **Automated P2P Order Processing:** Fetches pending Bybit P2P orders and initiates automated fiat payments via Paystack.
- **Bybit API Integration:** Handles fetching simplified and detailed P2P order information, including payment details. (Note: `confirm_bybit_payment` is a placeholder requiring implementation).
- **Paystack API Integration:** Manages bank code lookups, creates transfer recipients, initiates money transfers (NGN), and verifies transfer statuses.
- **Redis for State Management:**
    - Stores a historical log of all initiated transfers.
    - Implements a processing lock to prevent concurrent bot cycles.
    - Tracks "stuck" orders (e.g., due to general errors, insufficient funds, or pending OTP confirmation).
    - Caches Paystack transfer recipients for improved efficiency and reduced API calls.
- **Robust Error Handling & Logging:** Utilizes Python's `logging` module for comprehensive operational insights, warnings, and critical error reporting (to file and console). Includes specific handling for API request exceptions and Redis connectivity issues.
- **Scheduled Automation:** Employs `APScheduler` to run the bot's core processing cycle at configurable intervals, ensuring continuous operation.
- **Flask Dashboard API:** Provides a set of secure RESTful API endpoints for external monitoring and control:
    - `/api/status` (GET): Real-time operational health check of the bot and its dependencies.
    - `/api/orders/pending` (GET): Retrieves and displays current pending Bybit P2P orders.
    - `/api/transfers` (GET): Accesses the complete historical log of Paystack transfers.
    - `/api/control/start` (POST): Activates the bot's automated processing.
    - `/api/control/stop` (POST): Pauses the bot's automated processing.
    - `/api/cleanup/stuck-orders` (POST): Utility to manually clear stuck order flags and transfer history in Redis for reset/debugging.
- **Environment Variable Configuration:** Securely loads all sensitive API keys and configurable parameters from a `.env` file, promoting best practices for credential management.
- **Security:** Dashboard control endpoints are protected by an `X-Control-Secret` header, verified using HMAC for secure access.

## ⚙️ Architecture Overview
The system is composed of several key components working in concert:
- **`BybitP2PClient`**: A dedicated client class for making authenticated and signed requests to the Bybit P2P API.
- **`PaystackClient`**: A dedicated client class for interacting with the Paystack API, handling bank lookups, recipient management, and transfer operations. It includes recipient caching using Redis.
- **`P2PBotService`**: The central business logic class. It orchestrates the flow of fetching orders, determining their status, initiating payments, updating state in Redis, and handling various error conditions.
- **`Flask App`**: A lightweight web framework that exposes the monitoring and control API endpoints.
- **`BackgroundScheduler` (APScheduler)**: Manages the periodic execution of the `P2PBotService.run_cycle` method.
- **`Redis`**: An in-memory data store used for persistent state management, including transfer history, locks, and flags for stuck/insufficient funds orders.

## 🛠️ Getting Started

### Prerequisites
Before running the bot, ensure you have the following installed:
- **Python 3.8+**
- **Git** (for cloning the repository)
- A running **Redis server** (can be local or remote).

### Environment Variables (`.env` file)
Create a file named `.env` in the root directory of the project (same level as the Python script) and populate it with your API credentials and configuration settings. **Do NOT share this file publicly.**

Here's an example `.env.example` to guide you: