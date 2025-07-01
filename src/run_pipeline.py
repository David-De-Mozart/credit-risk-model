from data_processing import RFMTransformer, create_preprocessor, create_risk_labels
import pandas as pd
import joblib

# Load data
train = pd.read_csv('../data/raw/train.csv')

# Create RFM features
rfm_transformer = RFMTransformer(snapshot_date=pd.Timestamp('2023-01-01'))
rfm_features = rfm_transformer.transform(train)

# Create risk labels
rfm_labeled = create_risk_labels(rfm_features)

# Merge with original data
merged = train.merge(rfm_labeled, on='CustomerId', how='left')

# Save processed data
merged.to_csv('../data/processed/train_processed.csv', index=False)
print("Processed data saved with shape:", merged.shape)