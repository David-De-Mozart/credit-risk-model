from fastapi import FastAPI
from .pydantic_models import PredictionRequest, PredictionResponse
from src.data_processing import RFMTransformer, FeatureAggregator
import joblib
import pandas as pd
from pathlib import Path
import shap


app = FastAPI()

# Global variables for model and preprocessor
preprocessor = None
model = None


def load_artifacts():
    global preprocessor, model
    # Get project root directory
    project_dir = Path(__file__).resolve().parents[2]
    models_dir = project_dir / 'models'

    # Load preprocessor and model
    preprocessor = joblib.load(models_dir / 'preprocessor.joblib')
    model = joblib.load(models_dir / 'best_model.joblib')
    print("Model and preprocessor loaded successfully")


@app.on_event("startup")
async def startup_event():
    load_artifacts()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Convert to DataFrame
    transactions = pd.DataFrame([t.dict() for t in request.transactions])

    # Get customer ID
    customer_id = transactions['CustomerId'].iloc[0]

    # Calculate RFM features
    rfm_transformer = RFMTransformer()
    rfm_df = rfm_transformer.transform(transactions)

    # Aggregate features
    feature_agg = FeatureAggregator()
    agg_df = feature_agg.transform(transactions)

    # Merge features
    if 'CustomerId' in rfm_df.columns and 'CustomerId' in agg_df.columns:
        merged = rfm_df.merge(agg_df, on='CustomerId', how='left')
    else:
        # Fallback if CustomerId is missing
        merged = pd.concat([rfm_df, agg_df], axis=1)

    # Handle missing columns
    required_columns = [
        'Recency', 'Frequency', 'Monetary',
        'Amount_sum', 'Amount_mean', 'Amount_std', 'Amount_min', 'Amount_max',
        'Value_sum', 'Value_mean',
        'TransactionHour_mean', 'TransactionHour_std', 'TransactionDay_mean',
        'TransactionMonth_nunique',
        'ProductCategory_mode', 'ChannelId_mode'
    ]

    for col in required_columns:
        if col not in merged.columns:
            if col in ['ProductCategory_mode', 'ChannelId_mode']:
                merged[col] = 'missing'
            else:
                merged[col] = 0

    # Preprocess features
    features = merged[required_columns]
    X_processed = preprocessor.transform(features)

    # Make prediction
    risk_prob = model.predict_proba(X_processed)[0][1]

    # Calculate credit score (300-850 scale)
    credit_score = int(300 + (1 - risk_prob) * 550)

    # Determine risk category
    if risk_prob < 0.2:
        risk_category = "Low"
    elif risk_prob < 0.5:
        risk_category = "Medium"
    else:
        risk_category = "High"

    # Calculate recommended limit
    avg_transaction = transactions['Amount'].abs().mean()
    recommended_limit = max(500, avg_transaction * 10 * (1 - risk_prob))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    print(f"SHAP values: {shap_values[0]}")

    return PredictionResponse(
        customer_id=customer_id,
        risk_probability=risk_prob,
        risk_category=risk_category,
        credit_score=credit_score,
        recommended_limit=round(recommended_limit, 2)
    )
