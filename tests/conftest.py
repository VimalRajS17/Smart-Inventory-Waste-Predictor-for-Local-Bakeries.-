"""Pytest configuration, fixtures, and network isolation for Loaf Ledger."""

import os
import re
import pytest
import requests
from werkzeug.security import generate_password_hash

from app import create_app
from app.db import get_db, init_db


class MockResponse:
    """Mock requests.Response for Open-Meteo API tests."""

    def __init__(self, json_data, status_code=200, text=""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text or str(json_data)

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Enforce complete network isolation: fail if any test makes a real HTTP request."""
    def blocked_get(*args, **kwargs):
        raise RuntimeError("Real HTTP network calls are strictly forbidden during tests!")

    monkeypatch.setattr(requests, "get", blocked_get)


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Create an isolated test application instance with a fresh temporary SQLite database."""
    from app.auth import reset_rate_limits
    reset_rate_limits()
    test_db_path = str(tmp_path / "test_bakery.db")
    monkeypatch.setenv("DATABASE_PATH", test_db_path)
    monkeypatch.setenv("DEFAULT_LAT", "51.5074")
    monkeypatch.setenv("DEFAULT_LON", "-0.1278")
    monkeypatch.setenv("HOLIDAY_COUNTRY", "UK")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-12345")
    monkeypatch.setenv("DEMO_EMAIL", "demo@example.com")
    monkeypatch.setenv("DEMO_PASSWORD", "demo123")

    app_instance = create_app({
        "TESTING": True,
        "DATABASE": test_db_path,
        "DEFAULT_LAT": 51.5074,
        "DEFAULT_LON": -0.1278,
        "HOLIDAY_COUNTRY": "UK",
        "SECRET_KEY": "test-secret-key-12345",
    })

    with app_instance.app_context():
        init_db()
        db = get_db()

        # Seed default Demo User
        db.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            ("demo@example.com", generate_password_hash("demo123")),
        )
        demo_user_id = db.execute("SELECT id FROM users WHERE email = 'demo@example.com'").fetchone()["id"]

        # Seed default Demo Bakery shop
        db.execute(
            "INSERT INTO shops (owner_user_id, name, city, lat, lon) VALUES (?, ?, ?, ?, ?)",
            (demo_user_id, "Demo Bakery", "London", 51.5074, -0.1278),
        )
        shop_id = db.execute("SELECT id FROM shops WHERE owner_user_id = ? AND name = 'Demo Bakery'", (demo_user_id,)).fetchone()["id"]

        # Seed the standard items
        default_items = [
            ("Croissant", 20, 1),
            ("Sourdough Loaf", 15, 2),
            ("Chocolate Muffin", 25, 1),
            ("Cinnamon Roll", 15, 1),
            ("Cupcake", 20, 2),
        ]
        db.executemany(
            "INSERT INTO items (shop_id, name, current_stock, shelf_life_days) VALUES (?, ?, ?, ?)",
            [(shop_id, name, stock, shelf_life) for name, stock, shelf_life in default_items],
        )
        db.commit()

    return app_instance


class LoafTestClient:
    """Wrapped Flask test client with helper methods for CSRF and authentication."""

    def __init__(self, raw_client):
        self._raw_client = raw_client
        self._csrf_token = None

    def get_csrf(self):
        """Fetch CSRF token from /login or return cached token."""
        res = self._raw_client.get("/login")
        html = res.get_data(as_text=True)
        m = re.search(r'name="csrf_token" value="([^"]+)"', html) or re.search(r'name="csrf-token" content="([^"]+)"', html)
        if m:
            self._csrf_token = m.group(1)
        return self._csrf_token

    def _prepare_kwargs(self, method, kwargs):
        send_csrf = kwargs.pop("csrf", True)
        if send_csrf and method.upper() in ("POST", "PUT", "DELETE", "PATCH"):
            headers = kwargs.get("headers")
            if headers is None:
                headers = {}
                kwargs["headers"] = headers
            elif isinstance(headers, list):
                headers = dict(headers)
                kwargs["headers"] = headers
            if "X-CSRFToken" not in headers:
                token = self._csrf_token or self.get_csrf()
                if token:
                    headers["X-CSRFToken"] = token
        return kwargs

    def get(self, *args, **kwargs):
        return self._raw_client.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        kwargs = self._prepare_kwargs("POST", kwargs)
        return self._raw_client.post(*args, **kwargs)

    def put(self, *args, **kwargs):
        kwargs = self._prepare_kwargs("PUT", kwargs)
        return self._raw_client.put(*args, **kwargs)

    def delete(self, *args, **kwargs):
        kwargs = self._prepare_kwargs("DELETE", kwargs)
        return self._raw_client.delete(*args, **kwargs)

    def patch(self, *args, **kwargs):
        kwargs = self._prepare_kwargs("PATCH", kwargs)
        return self._raw_client.patch(*args, **kwargs)

    def session_transaction(self, *args, **kwargs):
        return self._raw_client.session_transaction(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._raw_client, name)


@pytest.fixture
def client(app):
    """Authenticated test client (logged in as demo@example.com) with auto CSRF."""
    c = LoafTestClient(app.test_client())
    token = c.get_csrf()
    c.post(
        "/login",
        data={"email": "demo@example.com", "password": "demo123", "csrf_token": token},
        csrf=False,
    )
    return c


@pytest.fixture
def unauth_client(app):
    """Unauthenticated test client."""
    return LoafTestClient(app.test_client())


@pytest.fixture
def runner(app):
    """Flask CLI test runner."""
    return app.test_cli_runner()


@pytest.fixture
def db_conn(app):
    """Database connection inside app context."""
    with app.app_context():
        yield get_db()
