"""Application factory for Loaf Ledger."""

import os
from datetime import date, timedelta
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required
from flask_wtf.csrf import CSRFError, CSRFProtect

from . import db
from app.services.weather import get_weather_for_date
from ml.predictor import predict_demand, predict_item_demand


def _load_env_fallback(env_path):
    """Fallback parser for .env files when python-dotenv is not installed."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k not in os.environ:
                os.environ[k] = v


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, ".env")

    # Load environment variables via python-dotenv if present, else fallback
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        _load_env_fallback(env_file)

    template_dir = os.path.join(project_root, "templates")
    static_dir = os.path.join(project_root, "static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
    )

    database_url = os.getenv("DATABASE_URL")
    db_path_env = os.getenv("DATABASE_PATH", "bakery.db")

    if database_url and database_url.startswith("sqlite:///"):
        sqlite_file = database_url[len("sqlite:///"):]
        if not os.path.isabs(sqlite_file):
            db_path = os.path.abspath(os.path.join(project_root, sqlite_file))
        else:
            db_path = sqlite_file
    elif not os.path.isabs(db_path_env):
        db_path = os.path.abspath(os.path.join(project_root, db_path_env))
    else:
        db_path = db_path_env

    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "0").lower() in ("1", "true", "yes"),
        REMEMBER_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_SAMESITE="Lax",
        REMEMBER_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "0").lower() in ("1", "true", "yes"),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,  # 2 MB request limit
        RATE_LIMIT_ENABLED=True,
        WTF_CSRF_TIME_LIMIT=None,
        DATABASE=db_path,
        DATABASE_URL=database_url,
        DEFAULT_LAT=float(os.getenv("DEFAULT_LAT", "51.5074")),
        DEFAULT_LON=float(os.getenv("DEFAULT_LON", "-0.1278")),
        HOLIDAY_COUNTRY=os.getenv("HOLIDAY_COUNTRY", "UK"),
        RL_ENABLED=os.getenv("RL_ENABLED", "0").lower() in ("1", "true", "yes"),
        DEBUG=os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes"),
    )

    if test_config is not None:
        app.config.from_mapping(test_config)

    # Configure structured logging
    import logging
    if not app.logger.handlers:
        log_handler = logging.StreamHandler()
        log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] in %(name)s: %(message)s")
        log_handler.setFormatter(log_formatter)
        app.logger.addHandler(log_handler)
    app.logger.setLevel(logging.INFO if not app.config.get("DEBUG") else logging.DEBUG)

    # Security Headers Middleware
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        return response

    # Error Handlers with Consistent JSON / Friendly HTML
    @app.errorhandler(400)
    def handle_bad_request(e):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "bad_request", "message": getattr(e, "description", "Bad request.")}}), 400
        return render_template("errors/400.html", error=getattr(e, "description", "Bad request.")), 400

    @app.errorhandler(404)
    def handle_not_found(e):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "not_found", "message": "The requested resource was not found."}}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "method_not_allowed", "message": "The method is not allowed for this URL."}}), 405
        return render_template("errors/404.html"), 405

    @app.errorhandler(413)
    def handle_payload_too_large(e):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "file_too_large", "message": "Request entity exceeds maximum allowed size (2 MB)."}}), 413
        return render_template("errors/400.html", error="Uploaded file exceeds maximum allowed size (2 MB)."), 413

    @app.errorhandler(429)
    def handle_rate_limited(e):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "rate_limited", "message": getattr(e, "description", "Too many requests. Please try again later.")}}), 429
        return render_template("auth/login.html", error=getattr(e, "description", "Too many attempts. Please try again later.")), 429

    @app.errorhandler(500)
    def handle_server_error(e):
        app.logger.exception("Internal Server Error: %s", str(e))
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "internal_server_error", "message": "An unexpected error occurred. Please try again later."}}), 500
        return render_template("errors/500.html"), 500

    # Initialize SQLite database handling and CLI commands
    db.init_app(app)

    # Initialize CSRF protection
    csrf = CSRFProtect(app)

    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "csrf_error", "message": e.description}}), 400
        return render_template("auth/csrf_error.html", reason=e.description), 400

    # Initialize Flask-Login
    from app.auth import User, auth_bp

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    @login_manager.unauthorized_handler
    def handle_unauthorized():
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"error": {"code": "unauthorized", "message": "Authentication required."}}), 401
        return redirect(url_for("auth.login", next=request.url))

    # Register blueprints
    app.register_blueprint(auth_bp)

    from app.api import api_bp
    app.register_blueprint(api_bp)

    @app.context_processor
    def inject_context():
        current_shop = None
        if current_user.is_authenticated:
            current_shop = current_user.get_shop()
        return {"current_shop": current_shop}

    # Register protected application page routes
    @app.route("/")
    @app.route("/plan")
    @login_required
    def plan_page():
        if not current_user.get_shop():
            return redirect(url_for("auth.onboarding"))
        return render_template("plan.html", active_tab="plan")

    @app.route("/today")
    @login_required
    def today_page():
        if not current_user.get_shop():
            return redirect(url_for("auth.onboarding"))
        return render_template("today.html", active_tab="today")

    @app.route("/reports")
    @login_required
    def reports_page():
        if not current_user.get_shop():
            return redirect(url_for("auth.onboarding"))
        return render_template("reports.html", active_tab="reports")

    @app.route("/items")
    @login_required
    def items_page():
        if not current_user.get_shop():
            return redirect(url_for("auth.onboarding"))
        return render_template("items.html", active_tab="items")

    @app.route("/import")
    @login_required
    def import_page():
        if not current_user.get_shop():
            return redirect(url_for("auth.onboarding"))
        return render_template("import.html", active_tab="import")

    @app.route("/settings")
    @login_required
    def settings_page():
        if not current_user.get_shop():
            return redirect(url_for("auth.onboarding"))
        return render_template("settings.html", active_tab="settings")

    # Legacy routes (kept for backwards compatibility)
    @app.route("/api/predict", methods=["GET"])
    @login_required
    def api_predict():
        """GET /api/predict?item=Croissant&date=2026-09-10&temperature_c=28&is_rainy=0"""
        item = request.args.get("item")
        target_date = request.args.get("date", (date.today() + timedelta(days=1)).isoformat())
        temperature_c = float(request.args.get("temperature_c", 25))
        is_rainy = int(request.args.get("is_rainy", 0))
        is_holiday = int(request.args.get("is_holiday", 0))

        if not item:
            return jsonify({"error": "item parameter is required"}), 400

        try:
            prediction = predict_demand(item, target_date, temperature_c, is_rainy, is_holiday)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

        return jsonify({
            "item": item,
            "date": target_date,
            "predicted_quantity": prediction,
        })

    @app.route("/log-sale", methods=["POST"])
    @login_required
    def log_sale():
        item_name = request.form["item_name"]
        quantity = int(request.form["quantity"])
        sale_date = request.form.get("date", date.today().isoformat())
        try:
            db.log_sale(item_name, sale_date, quantity)
            return jsonify({"status": "ok"})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/log-waste", methods=["POST"])
    @login_required
    def log_waste():
        item_name = request.form["item_name"]
        quantity = int(request.form["quantity"])
        reason = request.form.get("reason", "")
        waste_date = request.form.get("date", date.today().isoformat())
        try:
            db.log_waste(item_name, waste_date, quantity, reason)
            return jsonify({"status": "ok"})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/waste-summary-partial")
    @login_required
    def waste_summary_partial():
        """Returns waste-summary HTML fragment used by dashboard AJAX refresh."""
        waste_summary = db.get_waste_summary()
        return render_template("_waste_summary.html", waste_summary=waste_summary)

    # Initialize APScheduler if enabled
    try:
        from app.scheduler import init_scheduler
        init_scheduler(app)
    except Exception:
        pass

    return app
