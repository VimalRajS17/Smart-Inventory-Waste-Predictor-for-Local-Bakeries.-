"""Authentication and Onboarding blueprint for Loaf Ledger."""

from collections import defaultdict
import os
import time
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import UserMixin, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from app.db import get_db

auth_bp = Blueprint("auth", __name__)

_LOGIN_ATTEMPTS = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 10
LOGIN_WINDOW_SECONDS = 60


def _is_rate_limited(client_ip: str) -> bool:
    """Check if the client IP has exceeded the allowed login attempts in the sliding window."""
    now = time.time()
    attempts = [t for t in _LOGIN_ATTEMPTS[client_ip] if now - t < LOGIN_WINDOW_SECONDS]
    _LOGIN_ATTEMPTS[client_ip] = attempts
    return len(attempts) >= MAX_LOGIN_ATTEMPTS


def _record_login_failure(client_ip: str):
    """Record a failed login attempt timestamp for the client IP."""
    _LOGIN_ATTEMPTS[client_ip].append(time.time())


def _clear_login_attempts(client_ip: str):
    """Clear login attempts for the client IP after a successful login."""
    _LOGIN_ATTEMPTS.pop(client_ip, None)


def reset_rate_limits():
    """Reset rate limiter state (used by test fixtures)."""
    _LOGIN_ATTEMPTS.clear()


# ---------------------------------------------------------------------------
# User Model (Flask-Login UserMixin)
# ---------------------------------------------------------------------------

class User(UserMixin):
    """User representation for session authentication."""

    def __init__(self, id: int, email: str, password_hash: str, created_at=None):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at

    def check_password(self, password: str) -> bool:
        """Verify plain password against hashed password."""
        return check_password_hash(self.password_hash, password)

    def get_shop(self):
        """Fetch the shop owned by this user (or None if not yet onboarded)."""
        db = get_db()
        row = db.execute(
            "SELECT * FROM shops WHERE owner_user_id = ? ORDER BY id ASC LIMIT 1",
            (self.id,),
        ).fetchone()
        return dict(row) if row else None

    @classmethod
    def get(cls, user_id):
        """Load user by primary key ID."""
        try:
            uid = int(user_id)
        except (ValueError, TypeError):
            return None
        db = get_db()
        row = db.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE id = ?",
            (uid,),
        ).fetchone()
        if not row:
            return None
        return cls(row["id"], row["email"], row["password_hash"], row["created_at"])

    @classmethod
    def find_by_email(cls, email: str):
        """Lookup user by email address (case-insensitive)."""
        if not email or not isinstance(email, str):
            return None
        db = get_db()
        row = db.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE LOWER(email) = LOWER(?)",
            (email.strip(),),
        ).fetchone()
        if not row:
            return None
        return cls(row["id"], row["email"], row["password_hash"], row["created_at"])


# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """User registration route."""
    if current_user.is_authenticated:
        shop = current_user.get_shop()
        return redirect(url_for("plan_page") if shop else url_for("auth.onboarding"))

    if request.method == "POST":
        data = request.get_json(silent=True) if request.is_json else request.form
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        confirm = data.get("confirm_password") or ""

        # Validation
        if not email or "@" not in email or "." not in email:
            error_msg = "Please enter a valid email address."
            if request.is_json:
                return jsonify({"error": {"code": "invalid_input", "message": error_msg}}), 400
            return render_template("auth/register.html", error=error_msg, email=email), 400

        if len(password) < 6:
            error_msg = "Password must be at least 6 characters long."
            if request.is_json:
                return jsonify({"error": {"code": "invalid_input", "message": error_msg}}), 400
            return render_template("auth/register.html", error=error_msg, email=email), 400

        if password != confirm:
            error_msg = "Passwords do not match."
            if request.is_json:
                return jsonify({"error": {"code": "invalid_input", "message": error_msg}}), 400
            return render_template("auth/register.html", error=error_msg, email=email), 400

        db = get_db()
        existing = User.find_by_email(email)
        if existing:
            error_msg = "An account with this email address already exists."
            if request.is_json:
                return jsonify({"error": {"code": "duplicate_user", "message": error_msg}}), 400
            return render_template("auth/register.html", error=error_msg, email=email), 400

        # Create user
        cursor = db.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, generate_password_hash(password)),
        )
        db.commit()
        user_id = cursor.lastrowid
        new_user = User.get(user_id)

        # Log user in
        login_user(new_user)

        if request.is_json:
            return jsonify({
                "status": "ok",
                "user": {"id": new_user.id, "email": new_user.email},
                "redirect_url": url_for("auth.onboarding"),
            }), 201

        return redirect(url_for("auth.onboarding"))

    return render_template("auth/register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """User login route."""
    if current_user.is_authenticated:
        shop = current_user.get_shop()
        return redirect(url_for("plan_page") if shop else url_for("auth.onboarding"))

    if request.method == "POST":
        client_ip = request.remote_addr or "127.0.0.1"
        rate_limiting_enabled = current_app.config.get("RATE_LIMIT_ENABLED", True)
        if rate_limiting_enabled and _is_rate_limited(client_ip):
            error_msg = "Too many login attempts. Please wait a minute before trying again."
            if request.is_json:
                return jsonify({"error": {"code": "rate_limited", "message": error_msg}}), 429
            return render_template("auth/login.html", error=error_msg), 429

        data = request.get_json(silent=True) if request.is_json else request.form
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        remember = bool(data.get("remember"))

        user = User.find_by_email(email)
        if not user or not user.check_password(password):
            _record_login_failure(client_ip)
            error_msg = "Invalid email or password."
            if request.is_json:
                return jsonify({"error": {"code": "invalid_credentials", "message": error_msg}}), 401
            return render_template("auth/login.html", error=error_msg, email=email), 401

        _clear_login_attempts(client_ip)
        login_user(user, remember=remember)

        shop = user.get_shop()
        next_url = request.args.get("next") or (data.get("next") if not request.is_json else None)

        if not shop:
            target_url = url_for("auth.onboarding")
        elif next_url and next_url.startswith("/") and not next_url.startswith("//"):
            target_url = next_url
        else:
            target_url = url_for("plan_page")

        if request.is_json:
            return jsonify({
                "status": "ok",
                "user": {"id": user.id, "email": user.email},
                "redirect_url": target_url,
            }), 200

        return redirect(target_url)

    return render_template("auth/login.html")


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    """User logout route."""
    if current_user.is_authenticated:
        logout_user()
    if request.is_json:
        return jsonify({"status": "ok", "redirect_url": url_for("auth.login")}), 200
    return redirect(url_for("auth.login"))


# ---------------------------------------------------------------------------
# Onboarding Routes
# ---------------------------------------------------------------------------

@auth_bp.route("/onboarding", methods=["GET", "POST"])
@login_required
def onboarding():
    """Onboarding route: shop name, city, coordinates, and first items."""
    db = get_db()
    existing_shop = current_user.get_shop()
    if existing_shop and request.method == "GET":
        return redirect(url_for("plan_page"))

    if request.method == "POST":
        data = request.get_json(silent=True) if request.is_json else request.form
        name = (data.get("name") or data.get("shop_name") or "").strip()
        city = (data.get("city") or "").strip()
        lat_raw = data.get("lat") or data.get("latitude")
        lon_raw = data.get("lon") or data.get("longitude")

        if not name:
            error_msg = "Bakery shop name is required."
            if request.is_json:
                return jsonify({"error": {"code": "invalid_input", "message": error_msg}}), 400
            return render_template("auth/onboarding.html", error=error_msg), 400

        try:
            lat = float(lat_raw) if lat_raw is not None else float(current_app.config.get("DEFAULT_LAT", 51.5074))
            lon = float(lon_raw) if lon_raw is not None else float(current_app.config.get("DEFAULT_LON", -0.1278))
        except (ValueError, TypeError):
            error_msg = "Invalid latitude or longitude values."
            if request.is_json:
                return jsonify({"error": {"code": "invalid_input", "message": error_msg}}), 400
            return render_template("auth/onboarding.html", error=error_msg), 400

        # Upsert or create shop for current_user
        if existing_shop:
            db.execute(
                """
                UPDATE shops SET name = ?, city = ?, lat = ?, lon = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (name, city or None, lat, lon, existing_shop["id"], current_user.id),
            )
            shop_id = existing_shop["id"]
        else:
            cursor = db.execute(
                """
                INSERT INTO shops (owner_user_id, name, city, lat, lon)
                VALUES (?, ?, ?, ?, ?)
                """,
                (current_user.id, name, city or None, lat, lon),
            )
            shop_id = cursor.lastrowid
        db.commit()

        # Parse initial items
        items_created = 0
        if request.is_json:
            items_payload = data.get("items") or []
            for it in items_payload:
                it_name = (it.get("name") or "").strip()
                if not it_name:
                    continue
                stock = max(0, int(it.get("current_stock", 0)))
                shelf_life = max(1, int(it.get("shelf_life_days", 1)))
                db.execute(
                    """
                    INSERT INTO items (shop_id, name, current_stock, shelf_life_days)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(shop_id, name) DO NOTHING
                    """,
                    (shop_id, it_name, stock, shelf_life),
                )
                items_created += 1
        else:
            # HTML Form submission with item fields
            item_names = request.form.getlist("item_name[]")
            stocks = request.form.getlist("item_stock[]")
            shelf_lifes = request.form.getlist("item_shelf_life[]")

            for i in range(len(item_names)):
                it_name = item_names[i].strip() if i < len(item_names) else ""
                if not it_name:
                    continue
                try:
                    stock = max(0, int(stocks[i])) if i < len(stocks) else 0
                except (ValueError, TypeError):
                    stock = 0
                try:
                    shelf_life = max(1, int(shelf_lifes[i])) if i < len(shelf_lifes) else 1
                except (ValueError, TypeError):
                    shelf_life = 1

                db.execute(
                    """
                    INSERT INTO items (shop_id, name, current_stock, shelf_life_days)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(shop_id, name) DO NOTHING
                    """,
                    (shop_id, it_name, stock, shelf_life),
                )
                items_created += 1
        db.commit()

        # Fetch initial weather forecast
        try:
            from app.services.weather import fetch_forecast, save_weather
            forecast_rows = fetch_forecast(lat, lon, days=7)
            save_weather(shop_id, forecast_rows, "forecast")
        except Exception:
            pass

        if request.is_json:
            return jsonify({
                "status": "ok",
                "shop": {"id": shop_id, "name": name, "city": city, "lat": lat, "lon": lon},
                "items_created": items_created,
                "redirect_url": url_for("plan_page"),
            }), 201

        return redirect(url_for("plan_page"))

    default_lat = current_app.config.get("DEFAULT_LAT", 51.5074)
    default_lon = current_app.config.get("DEFAULT_LON", -0.1278)
    return render_template("auth/onboarding.html", default_lat=default_lat, default_lon=default_lon)

