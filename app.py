"""Entrypoint for running the Loaf Ledger Flask application."""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=app.config.get("DEBUG", False), port=5000)
