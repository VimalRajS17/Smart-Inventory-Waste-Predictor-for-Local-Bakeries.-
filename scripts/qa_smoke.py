#!/usr/bin/env python3
"""QA Smoke Test Script for Loaf Ledger.

Validates a live running instance against core health, security headers,
authentication, demand forecasting, accuracy reporting, simulation, and data export.

Usage:
    python scripts/qa_smoke.py [BASE_URL] [--email EMAIL] [--password PASSWORD]
    python scripts/qa_smoke.py http://127.0.0.1:5000
"""

import argparse
from datetime import date, timedelta
import re
import sys
import time
import requests


def run_smoke_checks(base_url: str, email: str, password: str) -> bool:
    """Execute smoke test suite against target base_url."""
    base_url = base_url.rstrip("/")
    session = requests.Session()
    print("=" * 80)
    print(f"LOAF LEDGER QA SMOKE TEST SUITE — TARGET: {base_url}")
    print("=" * 80)

    # Wait up to 10 seconds for server to be responsive
    server_ready = False
    for attempt in range(10):
        try:
            r = requests.get(f"{base_url}/api/health", timeout=2)
            if r.status_code == 200:
                server_ready = True
                break
        except Exception:
            time.sleep(1)

    if not server_ready:
        print(f"[FAIL] -> Server at {base_url} is not responding after 10s.")
        print("=" * 80)
        return False

    all_passed = True

    def check(name: str, fn):
        nonlocal all_passed
        print(f"[*] Testing {name} ... ", end="", flush=True)
        try:
            fn()
            print("[PASS]")
        except Exception as exc:
            all_passed = False
            print(f"[FAIL] -> {exc}")

    # 1. Health Endpoint
    def test_health():
        url = f"{base_url}/api/health"
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            raise AssertionError(f"Expected HTTP 200, got {res.status_code}")
        data = res.json()
        if data.get("status") != "ok":
            raise AssertionError(f"Expected status 'ok', got {data}")
    check("GET /api/health", test_health)

    # 2. Security Response Headers
    def test_security_headers():
        url = f"{base_url}/api/health"
        res = session.get(url, timeout=10)
        headers = res.headers
        if headers.get("X-Content-Type-Options") != "nosniff":
            raise AssertionError("Missing X-Content-Type-Options: nosniff")
        if headers.get("X-Frame-Options") != "SAMEORIGIN":
            raise AssertionError("Missing X-Frame-Options: SAMEORIGIN")
        if "Content-Security-Policy" not in headers:
            raise AssertionError("Missing Content-Security-Policy header")
    check("Security Headers (X-Content-Type-Options, X-Frame-Options, CSP)", test_security_headers)

    # 3. Authentication & Login
    def test_login():
        login_page_url = f"{base_url}/login"
        page_res = session.get(login_page_url, timeout=10)
        if page_res.status_code != 200:
            raise AssertionError(f"Could not load login page (HTTP {page_res.status_code})")

        # Extract CSRF token
        html = page_res.text
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', html) or re.search(r'name="csrf-token"\s+content="([^"]+)"', html)
        csrf_token = m.group(1) if m else ""

        login_url = f"{base_url}/login"
        payload = {
            "email": email,
            "password": password,
            "csrf_token": csrf_token,
        }
        res = session.post(login_url, data=payload, headers={"Referer": login_page_url}, timeout=10, allow_redirects=True)
        if res.status_code not in (200, 302):
            raise AssertionError(f"Login failed with HTTP {res.status_code}")
        # Verify authenticated access to /api/items
        items_res = session.get(f"{base_url}/api/items", timeout=10)
        if items_res.status_code != 200:
            raise AssertionError(f"Post-login verification on /api/items failed (HTTP {items_res.status_code})")
    check(f"Authentication (login as {email})", test_login)

    # 4. Multi-item Forecast Generation
    def test_forecast():
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        url = f"{base_url}/api/forecast?date={tomorrow}"
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            raise AssertionError(f"Expected HTTP 200, got {res.status_code}")
        data = res.json()
        if "forecasts" not in data or not isinstance(data["forecasts"], list):
            raise AssertionError("Expected 'forecasts' array in response")
        if len(data["forecasts"]) == 0:
            raise AssertionError("Forecasts array is unexpectedly empty")
    check("GET /api/forecast (tomorrow demand forecast)", test_forecast)

    # 5. Accuracy Telemetry Report
    def test_accuracy_report():
        url = f"{base_url}/api/reports/accuracy?days=28"
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            raise AssertionError(f"Expected HTTP 200, got {res.status_code}")
        data = res.json()
        if data.get("status") != "success":
            raise AssertionError(f"Expected status 'success', got {data.get('status')}")
        if "items" not in data or "overall" not in data:
            raise AssertionError("Missing 'items' or 'overall' in accuracy response")
    check("GET /api/reports/accuracy (28-day accuracy telemetry)", test_accuracy_report)

    # 6. Simulation on Past Data
    def test_simulation_report():
        url = f"{base_url}/api/reports/simulation?days=60"
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            raise AssertionError(f"Expected HTTP 200, got {res.status_code}")
        data = res.json()
        if data.get("simulation_label") != "Simulation on past data":
            raise AssertionError(f"Expected simulation_label 'Simulation on past data', got '{data.get('simulation_label')}'")
        if "policies" not in data or len(data["policies"]) == 0:
            raise AssertionError("Missing or empty policies in simulation response")
    check("GET /api/reports/simulation ('Simulation on past data')", test_simulation_report)

    # 7. CSV Export
    def test_csv_export():
        url = f"{base_url}/api/export/daily-log"
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            raise AssertionError(f"Expected HTTP 200, got {res.status_code}")
        if "text/csv" not in res.headers.get("Content-Type", ""):
            raise AssertionError(f"Expected text/csv Content-Type, got '{res.headers.get('Content-Type')}'")
        lines = res.text.strip().splitlines()
        if len(lines) < 1 or "date,item_name" not in lines[0]:
            raise AssertionError("CSV header does not match expected daily log format")
    check("GET /api/export/daily-log (RFC 4180 CSV export)", test_csv_export)

    print("=" * 80)
    if all_passed:
        print("ALL QA SMOKE CHECKS PASSED SUCCESSFULLY (0 failures).")
        print("=" * 80)
        return True
    else:
        print("QA SMOKE TEST SUITE FAILED WITH ONE OR MORE ERRORS.")
        print("=" * 80)
        return False


def main():
    parser = argparse.ArgumentParser(description="Loaf Ledger QA Smoke Test Runner")
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:5000", help="Base URL of running application (default: http://127.0.0.1:5000)")
    parser.add_argument("--email", default="demo@example.com", help="Login email (default: demo@example.com)")
    parser.add_argument("--password", default="demo123", help="Login password (default: demo123)")
    args = parser.parse_args()

    success = run_smoke_checks(args.url, args.email, args.password)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
