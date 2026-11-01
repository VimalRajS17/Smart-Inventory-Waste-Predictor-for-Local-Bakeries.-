"""API blueprint for Loaf Ledger with unified error formatting."""

from flask import Blueprint, jsonify

api_bp = Blueprint("api", __name__, url_prefix="/api")


def error_response(code: str, message: str, status_code: int = 400):
    """Generate consistent JSON error response shape: {"error": {"code": "...", "message": "..."}}."""
    return jsonify({"error": {"code": code, "message": message}}), status_code


@api_bp.errorhandler(400)
def handle_bad_request(e):
    from flask_wtf.csrf import CSRFError
    if isinstance(e, CSRFError):
        return error_response("csrf_error", str(getattr(e, "description", "CSRF token missing or invalid.")), 400)
    msg = getattr(e, "description", "Bad request.")
    return error_response("bad_request", str(msg), 400)


@api_bp.errorhandler(404)
def handle_not_found(e):
    msg = getattr(e, "description", "Resource not found.")
    return error_response("not_found", str(msg), 404)


@api_bp.errorhandler(405)
def handle_method_not_allowed(e):
    return error_response("method_not_allowed", "Method not allowed for this endpoint.", 405)


@api_bp.errorhandler(500)
def handle_internal_error(e):
    return error_response("server_error", "An unexpected server error occurred.", 500)


# Import routes to register endpoints on api_bp
from . import routes  # noqa: E402, F401

