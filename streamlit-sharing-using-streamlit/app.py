import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------
# Load the four persisted artifacts ONCE. Streamlit re-runs this whole
# script top-to-bottom on every widget interaction, so @st.cache_resource
# keeps joblib.load() from re-reading the files from disk on every click.
# MODEL_DIR is found relative to this file, so the app works whether it is
# started from the repo root, from this folder, or on Streamlit Cloud.
# -----------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")


@st.cache_resource
def load_artifact(file_name):
    """Load one saved model file once. Returns None if the file is not in model/ yet,
    so one missing model does not crash the whole app."""
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# Each model owner's saved file (change a name here when an owner's model is merged).
REGRESSION_FILE = "ridge_regressor_for_farm_income.joblib"          # Member 2
CLASSIFICATION_FILE = "invoice_payment_risk_classifier.joblib"      # Member 3
CLUSTERING_FILE = "kmeans_farmer_segmentation.joblib"               # Member 4 (Maxwell)
RECOMMENDER_FILE = "apriori_recommendation_rules.joblib"            # Member 5 -> replace with our food recommender

regression_pipeline = load_artifact(REGRESSION_FILE)
classification_pipeline = load_artifact(CLASSIFICATION_FILE)
clustering_artifact = load_artifact(CLUSTERING_FILE)
arm_artifact = load_artifact(RECOMMENDER_FILE)


def model_missing(file_name):
    st.warning(f"Model file `model/{file_name}` is not in the repository yet. "
               "This tab will work once its owner's pull request is merged.")


st.set_page_config(page_title="BBT 4206 Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("BBT 4206 -- Model Validation Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "Farm Income Regressor", "Invoice Payment Risk Classifier",
    "Farmer Segmenter", "E-commerce Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression -- annual farm income in KES (Member 2)
# -----------------------------------------------------------------------
with tab_regression:
    st.caption("Predicts a Kenyan farm's annual income (KES).")
    with st.form("regression_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            # The pipeline's OneHotEncoder was fitted on exactly these categories.
            crop_type = st.selectbox(
                "Crop Type", ["Coffee", "Dairy", "Horticulture", "Maize", "Tea"], index=4, key="reg_crop")
            county = st.selectbox(
                "County", ["Kericho", "Kiambu", "Meru", "Nakuru", "Uasin Gishu", "Other"], key="reg_county")
            irrigation_type = st.selectbox(
                "Irrigation Type", ["Manual", "Drip", "Automated", "None"], key="reg_irrigation")
            cooperative_member = st.selectbox("Cooperative Member", ["No", "Yes"], key="reg_coop")
            farm_size_acres = st.number_input(
                "Farm Size (acres)", min_value=0.0, value=10.49, step=0.1, key="reg_size")
            years_farming_experience = st.number_input(
                "Years of Farming Experience", min_value=0.0, value=9.3, step=0.1, key="reg_exp")
        with col_b:
            num_workers_employed = st.number_input(
                "Workers Employed", min_value=0, value=1, step=1, key="reg_workers")
            rainfall_mm = st.number_input(
                "Rainfall (mm per year)", min_value=0.0, value=967.0, step=10.0, key="reg_rain")
            soil_quality_score = st.number_input(
                "Soil Quality Score", min_value=0.0, value=54.5, step=0.1, key="reg_soil")
            market_distance_km = st.number_input(
                "Distance to Market (km)", min_value=0.0, value=6.0, step=0.1, key="reg_distance")
            irrigation_maintenance_cost_kes = st.number_input(
                "Irrigation Maintenance Cost (KES, ignored if no irrigation)",
                min_value=0.0, value=3207.36, step=100.0, key="reg_maint")
        submitted = st.form_submit_button("Predict Farm Income")

    if submitted and regression_pipeline is None:
        model_missing(REGRESSION_FILE)
    elif submitted:
        payload = {
            "crop_type": crop_type, "county": county,
            "irrigation_type": irrigation_type, "cooperative_member": cooperative_member,
            "farm_size_acres": farm_size_acres,
            "years_farming_experience": years_farming_experience,
            "num_workers_employed": num_workers_employed,
            "rainfall_mm": rainfall_mm, "soil_quality_score": soil_quality_score,
            "market_distance_km": market_distance_km,
            # No irrigation system means no maintenance cost: send it as missing,
            # exactly like the API does, and the pipeline's imputer fills it.
            "irrigation_maintenance_cost_kes": (
                None if irrigation_type == "None" else irrigation_maintenance_cost_kes),
        }
        input_row = pd.DataFrame([payload]).astype({"irrigation_maintenance_cost_kes": float})
        # The pipeline already converts back from log to KES (TransformedTargetRegressor),
        # so no np.expm1 here.
        predicted_income_kes = float(regression_pipeline.predict(input_row)[0])
        st.success(f"Predicted annual farm income: KES {predicted_income_kes:,.2f}")

# -----------------------------------------------------------------------
# 2. Classification -- invoice payment risk (Member 3)
# -----------------------------------------------------------------------
with tab_classification:
    st.caption("Predicts whether a B2B invoice will be paid On-Time, Late or Severely Late.")
    with st.form("classification_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            client_industry = st.selectbox(
                "Client Industry",
                ["Technology", "Manufacturing", "Retail", "Construction", "Healthcare"],
                key="clf_industry")
            payment_method = st.selectbox(
                "Payment Method",
                ["Bank Transfer", "Mobile Money", "Cheque", "Letter of Credit"],
                key="clf_method")
            payment_terms_days = st.number_input(
                "Payment Terms (days)", min_value=0, value=30, step=1, key="clf_terms")
            prior_late_payments_count = st.number_input(
                "Prior Late Payments", min_value=0, value=1, step=1, key="clf_prior_late")
            days_outstanding_avg_historical = st.number_input(
                "Avg Days Outstanding (history)", min_value=0.0, value=16.4, step=0.1,
                key="clf_days_out")
        with col_b:
            # The pipeline's OrdinalEncoder was fitted on exactly these four
            # ratings, in this order (Poor < Fair < Good < Excellent).
            client_credit_rating = st.selectbox(
                "Client Credit Rating", ["Poor", "Fair", "Good", "Excellent"], index=2,
                key="clf_rating")
            has_dispute_history = st.selectbox(
                "Has Dispute History", ["No", "Yes"], key="clf_dispute")
            invoice_amount_kes = st.number_input(
                "Invoice Amount (KES)", min_value=0.0, value=371672.0, step=1000.0,
                key="clf_amount")
            discount_offered_pct = st.number_input(
                "Discount Offered (%)", min_value=0.0, value=2.06, step=0.01, key="clf_discount")
            economic_sector_growth_rate_pct = st.number_input(
                "Sector Growth Rate (%)", value=4.42, step=0.01, key="clf_growth")
        submitted_clf = st.form_submit_button("Predict Payment Risk")

    if submitted_clf and classification_pipeline is None:
        model_missing(CLASSIFICATION_FILE)
    elif submitted_clf:
        # Exactly the column names the pipeline was trained on.
        payload = {
            "client_industry": client_industry,
            "client_credit_rating": client_credit_rating,
            "payment_method": payment_method,
            "has_dispute_history": has_dispute_history,
            "payment_terms_days": payment_terms_days,
            "invoice_amount_kes": invoice_amount_kes,
            "prior_late_payments_count": prior_late_payments_count,
            "discount_offered_pct": discount_offered_pct,
            "days_outstanding_avg_historical": days_outstanding_avg_historical,
            "economic_sector_growth_rate_pct": economic_sector_growth_rate_pct,
        }
        input_row = pd.DataFrame([payload])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
        st.success(f"Predicted payment risk: {predicted_class}")
        for label, prob in sorted(zip(class_labels, predicted_probabilities),
                                  key=lambda item: item[1], reverse=True):
            st.progress(float(prob), text=f"{label}: {prob:.1%}")

# -----------------------------------------------------------------------
# 3. Clustering -- farmer segmentation (Member 4: Maxwell)
# -----------------------------------------------------------------------
with tab_clustering:
    st.caption("Assigns a farmer to one of 5 segments found by K-Means clustering.")
    if clustering_artifact is None:
        model_missing(CLUSTERING_FILE)
    else:
        clustering_scaler = clustering_artifact["scaler"]
        clustering_kmeans = clustering_artifact["kmeans"]
        with st.form("clustering_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                farm_size_acres = st.number_input(
                    "Farm Size (acres)", min_value=0.0, value=16.29, step=0.1, key="clu_size")
                annual_yield_tonnes = st.number_input(
                    "Annual Yield (tonnes)", min_value=0.0, value=35.9, step=0.1, key="clu_yield")
                income_per_acre_kes = st.number_input(
                    "Income per Acre (KES)", min_value=0.0, value=2000.0, step=100.0, key="clu_income")
            with col_b:
                years_farming_experience = st.number_input(
                    "Years of Farming Experience", min_value=0.0, value=7.0, step=1.0, key="clu_exp")
                market_access_score = st.number_input(
                    "Market Access Score", min_value=0.0, value=51.72, step=0.1, key="clu_market")
            submitted3 = st.form_submit_button("Predict Segment")

        if submitted3:
            # Exactly the column names the scaler was fitted on.
            input_row = pd.DataFrame([{
                "farm_size_acres": farm_size_acres,
                "annual_yield_tonnes": annual_yield_tonnes,
                "income_per_acre_kes": income_per_acre_kes,
                "years_farming_experience": years_farming_experience,
                "market_access_score": market_access_score,
            }])[list(clustering_scaler.feature_names_in_)]
            cluster = int(clustering_kmeans.predict(clustering_scaler.transform(input_row))[0])
            st.success(f"Assigned segment: Cluster {cluster}")

            # What a typical farmer in each segment looks like (cluster centres in real units).
            centres = pd.DataFrame(
                clustering_scaler.inverse_transform(clustering_kmeans.cluster_centers_),
                columns=clustering_scaler.feature_names_in_,
            ).round(2)
            centres.index.name = "Cluster"
            st.write("Typical farmer in each segment (this farmer's segment is highlighted):")
            st.dataframe(centres.style.apply(
                lambda r: ["background-color: #d4edda" if r.name == cluster else "" for _ in r], axis=1))

# -----------------------------------------------------------------------
# 4. Association Rule Mining
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        cart_text = st.text_input("Cart (comma-separated SKUs)", "H06, H07")
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4 and arm_artifact is None:
        model_missing(RECOMMENDER_FILE)
    elif submitted4:
        recommendation_lookup = arm_artifact["recommendation_lookup"]
        sku_lookup = arm_artifact["sku_lookup"]
        cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
        candidates = {}
        for antecedent, consequent_list in recommendation_lookup.items():
            if antecedent.issubset(cart):
                for sku, confidence, lift in consequent_list:
                    if sku not in cart:
                        if sku not in candidates or confidence > candidates[sku][0]:
                            candidates[sku] = (confidence, lift)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
        if not ranked:
            st.warning("No recommendations for this cart -- try a SKU combination seen in the training orders.")
        else:
            for sku, (confidence, lift) in ranked:
                st.write(f"**{sku_lookup.loc[sku, 'product_name']}** ({sku}) "
                         f"-- confidence {confidence:.1%}, lift {lift:.2f}")