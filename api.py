"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
trained by our group and loaded from the model/ folder (never retrained here):

    1. Regression       -- annual farm income (KES) for a farm
    2. Classification   -- invoice payment risk (On-Time / Late / Severely Late)
    3. Clustering       -- farmer segment assignment
    4. Association Rule -- food item recommendation from an order cart
       Mining (ARM)

Each fitted scikit-learn / imbalanced-learn Pipeline is a SINGLE object that
already knows how to impute, encode and scale, so this file builds a DataFrame
straight from the incoming JSON and hands it to the saved model as-is.
"The pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "ridge_regressor_for_farm_income.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "invoice_payment_risk_classifier.joblib")
)

farmer_clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_farmer_segmentation.joblib")
)
farmer_scaler = farmer_clustering_artifact["scaler"]
farmer_kmeans = farmer_clustering_artifact["kmeans"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# 1. REGRESSION -- predict annual farm income (KES) for a Kenyan farm

# Sample request (PowerShell):
#   Invoke-RestMethod -Method Post `
#     -Uri "http://127.0.0.1:5000/api/v1/models/farm-income-regressor/predictions" `
#     -ContentType "application/json" `
#     -Body (@{crop_type="Tea"; county="Kericho"; irrigation_type="Manual";
#              cooperative_member="No"; farm_size_acres=10.49;
#              years_farming_experience=9.3; num_workers_employed=1;
#              rainfall_mm=967.0; soil_quality_score=54.5;
#              market_distance_km=6.0;
#              irrigation_maintenance_cost_kes=3207.36} | ConvertTo-Json)
#
# Note: irrigation_maintenance_cost_kes can be null when irrigation_type is
# "None" (no irrigation system means no maintenance cost). The pipeline's own
# imputer fills it with 0; nothing in this file needs to know that rule.
@app.route("/api/v1/models/farm-income-regressor/predictions", methods=["POST"])
def predict_farm_income():
    payload = request.get_json()
    # A JSON null arrives as None, which makes pandas type this column as
    # "object"; cast it to float so None becomes NaN and the pipeline's
    # constant-0 imputer recognises it as missing.
    input_row = pd.DataFrame([payload]).astype(
        {"irrigation_maintenance_cost_kes": float}
    )

    # The pipeline wraps Ridge in a TransformedTargetRegressor
    # (log1p for training, expm1 for prediction), so .predict() already
    # returns KES. Do NOT apply np.expm1 again here.
    predicted_income_kes = float(regression_pipeline.predict(input_row)[0])

    return jsonify({
        "predicted_annual_farm_income_kes": round(predicted_income_kes, 2),
        "model": "Ridge Regressor (tuned, alpha=0.01)",
    })

# 2. CLASSIFICATION -- predict invoice payment risk (On-Time / Late / Severely Late)
# Sample request (cURL) -- invoice from invoice_payment_risk.csv
# (invoice_id 590, actual category: On-Time):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/invoice-payment-risk-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"client_industry": "Technology", "client_credit_rating": "Good",
#             "payment_method": "Bank Transfer", "has_dispute_history": "No",
#             "payment_terms_days": 30, "invoice_amount_kes": 371672.0,
#             "prior_late_payments_count": 1, "discount_offered_pct": 2.06,
#             "days_outstanding_avg_historical": 16.4,
#             "economic_sector_growth_rate_pct": 4.42}'

INVOICE_NUMERIC_FIELDS = [
    "payment_terms_days", "invoice_amount_kes", "prior_late_payments_count",
    "discount_offered_pct", "days_outstanding_avg_historical",
    "economic_sector_growth_rate_pct",
]
INVOICE_CATEGORICAL_FIELDS = [
    "client_industry", "client_credit_rating", "payment_method",
    "has_dispute_history",
]
INVOICE_CREDIT_RATINGS = ["Poor", "Fair", "Good", "Excellent"]

@app.route("/api/v1/models/invoice-payment-risk-classifier/predictions", methods=["POST"])
def predict_payment_risk():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "The request body must be a JSON object."}), 400

    missing = [
        field for field in INVOICE_NUMERIC_FIELDS + INVOICE_CATEGORICAL_FIELDS
        if payload.get(field) is None or payload.get(field) == ""
    ]
    if missing:
        return jsonify({
            "error": "Missing required field(s): " + ", ".join(missing),
            "missing_fields": missing,
        }), 400

    row = {}
    for field in INVOICE_NUMERIC_FIELDS:
        try:
            row[field] = float(payload[field])
        except (TypeError, ValueError):
            return jsonify({"error": f"Field '{field}' must be a number."}), 400
    for field in INVOICE_CATEGORICAL_FIELDS:
        row[field] = str(payload[field])

    if row["client_credit_rating"] not in INVOICE_CREDIT_RATINGS:
        return jsonify({
            "error": "client_credit_rating must be one of: "
                     + ", ".join(INVOICE_CREDIT_RATINGS),
        }), 400

    input_row = pd.DataFrame([row])

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_payment_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Logistic Regression (tuned, L1, C=10; SMOTE-resampled at training only)",
    })


# 3. CLUSTERING -- assign a new farmer to a segment
#
# Sample request (PowerShell):
#   Invoke-RestMethod -Method Post `
#     -Uri "http://127.0.0.1:5000/api/v1/models/farmer-segmenter/predictions" `
#     -ContentType "application/json" `
#     -Body '{"farm_size_acres": 16.29, "annual_yield_tonnes": 35.9,
#             "income_per_acre_kes": 2000, "years_farming_experience": 7,
#             "market_access_score": 51.72}'
@app.route("/api/v1/models/farmer-segmenter/predictions", methods=["POST"])
def predict_farmer_segment():
    payload = request.get_json(silent=True) or {}
    expected = list(farmer_scaler.feature_names_in_)
    missing = [col for col in expected if col not in payload]
    if missing:
        return jsonify({"error": "Missing input field(s)", "missing": missing,
                        "expected_fields": expected}), 400

    input_row = pd.DataFrame([payload])[expected]
    scaled_input = farmer_scaler.transform(input_row)
    assigned_cluster = int(farmer_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": f"K-Means (k={farmer_kmeans.n_clusters}, standardized features)",
    })

# 4. ASSOCIATION RULE MINING -- recommend food items for an order cart
#
# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["F10", "F12"]}'
#
# Note: there is no scikit-learn ".predict()" here -- the mined association
# rules were precomputed into a lookup dictionary (antecedent item-set ->
# ranked candidate consequents) in the notebook. This endpoint only looks up
# and ranks; it does not mine for the rules again. Bad input returns a clear
# 400 JSON error.
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    cart = payload.get("cart")
    top_n = payload.get("top_n", 3)

    if not isinstance(cart, list) or not cart:
        return jsonify({
            "error": "cart must be a non-empty list of food item IDs."
        }), 400

    if not all(isinstance(item, str) for item in cart):
        return jsonify({
            "error": "Every cart item must be a string."
        }), 400

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        return jsonify({
            "error": "top_n must be a positive integer."
        }), 400

    unknown_items = sorted(set(cart) - set(sku_lookup.index))

    if unknown_items:
        return jsonify({
            "error": "Unknown food item ID(s).",
            "unknown_items": unknown_items
        }), 400

    cart_set = set(cart)
    candidates = {}

    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart_set):
            for sku, confidence, lift in consequent_list:
                if sku not in cart_set:
                    if (
                        sku not in candidates
                        or confidence > candidates[sku][0]
                    ):
                        candidates[sku] = (confidence, lift)

    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1][0],
        reverse=True
    )[:top_n]

    recommendations = [
        {
            "sku": str(sku),
            "product_name": str(sku_lookup.loc[sku, "product_name"]),
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2)
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart_set),
        "recommendations": recommendations,
        "model": "Food Delivery Apriori Association Rules"
    })



# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
