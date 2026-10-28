from datetime import date, timedelta

from flask import Flask, jsonify, render_template, request

import database
from models.predictor import predict_demand

app = Flask(__name__)
database.init_db()


@app.route("/")
def dashboard():
    items = database.get_all_items()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    predictions = []
    for item in items:
        try:
            predicted_qty = predict_demand(item["name"], tomorrow)
        except ValueError:
            predicted_qty = None
        predictions.append({
            "name": item["name"],
            "current_stock": item["current_stock"],
            "predicted_demand": predicted_qty,
            "suggested_order": (
                max(0, round(predicted_qty - item["current_stock"], 1))
                if predicted_qty is not None else None
            ),
        })

    waste_summary = database.get_waste_summary()
    return render_template(
        "dashboard.html",
        predictions=predictions,
        waste_summary=waste_summary,
        target_date=tomorrow,
        item_names=[item["name"] for item in items],
    )


@app.route("/api/predict", methods=["GET"])
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
def log_sale():
    item_name = request.form["item_name"]
    quantity = int(request.form["quantity"])
    sale_date = request.form.get("date", date.today().isoformat())
    database.log_sale(item_name, sale_date, quantity)
    return jsonify({"status": "ok"})


@app.route("/log-waste", methods=["POST"])
def log_waste():
    item_name = request.form["item_name"]
    quantity = int(request.form["quantity"])
    reason = request.form.get("reason", "")
    waste_date = request.form.get("date", date.today().isoformat())
    database.log_waste(item_name, waste_date, quantity, reason)
    return jsonify({"status": "ok"})


@app.route("/waste-summary-partial")
def waste_summary_partial():
    """Returns just the waste-summary HTML fragment, used by the dashboard's
    AJAX refresh after logging waste — avoids a full page reload."""
    waste_summary = database.get_waste_summary()
    return render_template("_waste_summary.html", waste_summary=waste_summary)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
