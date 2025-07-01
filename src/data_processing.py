import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Set random state for reproducibility
RANDOM_STATE = 42

# Define mode function


def get_mode(x):
    if x.empty:
        return 'missing'
    return x.mode()[0]


class RFMTransformer(BaseEstimator, TransformerMixin):
    """Calculates RFM features from transaction data"""

    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date or pd.Timestamp.now()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to datetime
        if 'TransactionStartTime' not in X.columns:
            raise ValueError(
                "DataFrame is missing 'TransactionStartTime' column")

        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])

        # Calculate RFM
        recency = X.groupby('CustomerId')['TransactionStartTime'].max().apply(
            lambda x: (self.snapshot_date - x).days
        ).reset_index(name='Recency')

        frequency = X.groupby('CustomerId').size(
        ).reset_index(name='Frequency')

        # Only consider positive amounts for monetary value
        monetary = X[X['Amount'] > 0].groupby(
            'CustomerId')['Amount'].sum().reset_index(name='Monetary')

        # Merge features
        rfm = recency.merge(
            frequency,
            on='CustomerId').merge(
            monetary,
            on='CustomerId',
            how='left')
        rfm.fillna(0, inplace=True)

        return rfm


class FeatureAggregator(BaseEstimator, TransformerMixin):
    """Aggregates transaction-level features to customer level"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to datetime
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])

        # Extract temporal features
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year

        # Aggregation dictionary
        agg_funcs = {
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'Value': ['sum', 'mean'],
            'ProductCategory': ['nunique', get_mode],
            'ChannelId': ['nunique', get_mode],
            'TransactionHour': ['mean', 'std'],
            'TransactionDay': ['mean'],
            'TransactionMonth': ['nunique'],
            'TransactionYear': ['max']
        }

        # Group by customer
        agg_df = X.groupby('CustomerId').agg(agg_funcs)

        # Flatten multi-level columns
        agg_df.columns = ['_'.join(col).strip()
                          for col in agg_df.columns.values]
        agg_df.reset_index(inplace=True)

        # Rename mode columns for clarity
        agg_df.columns = [col.replace('get_mode', 'mode')
                          for col in agg_df.columns]

        return agg_df


def create_proxy_target(rfm_df):
    """Creates high-risk labels using K-Means clustering"""
    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Cluster customers
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(scaled)
    rfm_df['Cluster'] = clusters

    # Identify high-risk cluster (high recency, low frequency/monetary)
    cluster_means = rfm_df.groupby(
        'Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    cluster_means['RiskScore'] = (
        cluster_means['Recency'] * 0.6 -
        cluster_means['Frequency'] * 0.2 -
        cluster_means['Monetary'] * 0.2
    )
    high_risk_cluster = cluster_means['RiskScore'].idxmax()

    # Create binary label
    rfm_df['is_high_risk'] = (
        rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df.drop(columns=['Cluster'])


def build_preprocessor():
    """Builds data preprocessing pipeline"""
    # Numeric features
    numeric_features = [
        'Recency', 'Frequency', 'Monetary',
        'Amount_sum', 'Amount_mean', 'Amount_std', 'Amount_min', 'Amount_max',
        'Value_sum', 'Value_mean',
        'TransactionHour_mean', 'TransactionHour_std', 'TransactionDay_mean',
        'TransactionMonth_nunique'
    ]

    # Categorical features
    categorical_features = [
        'ProductCategory_mode',
        'ChannelId_mode'
    ]

    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def run_pipeline():
    """Main data processing pipeline"""
    print("Starting data processing pipeline...")

    # Get project root directory
    project_dir = Path(__file__).resolve().parents[1]

    # Create directories
    processed_dir = project_dir / 'data' / 'processed'
    models_dir = project_dir / 'models'
    raw_dir = project_dir / 'data' / 'raw'

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Verify files exist
    train_path = raw_dir / 'train.csv'
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")

    # Load data
    print("Loading data...")
    train = pd.read_csv(train_path)
    print(f"Original data shape: {train.shape}")

    # Set snapshot date (most recent transaction date)
    print("Calculating snapshot date...")
    train['TransactionStartTime'] = pd.to_datetime(
        train['TransactionStartTime'])
    snapshot_date = train['TransactionStartTime'].max()
    print(f"Snapshot date: {snapshot_date}")

    # Calculate RFM features
    print("Calculating RFM features...")
    rfm_transformer = RFMTransformer(snapshot_date)
    rfm_df = rfm_transformer.transform(train)
    print(f"RFM features shape: {rfm_df.shape}")

    # Create proxy target
    print("Creating proxy target...")
    rfm_df = create_proxy_target(rfm_df)
    print("Target distribution:")
    print(rfm_df['is_high_risk'].value_counts(normalize=True))

    # Aggregate features
    print("Aggregating features...")
    feature_agg = FeatureAggregator()
    agg_df = feature_agg.transform(train)
    print(f"Aggregated features shape: {agg_df.shape}")
    print("Sample aggregated columns:", agg_df.columns[:5])

    # Merge datasets
    print("Merging datasets...")
    merged = rfm_df.merge(agg_df, on='CustomerId', how='left')
    print(f"Merged data shape: {merged.shape}")

    # Save processed data
    processed_csv_path = processed_dir / 'train_processed.csv'
    merged.to_csv(processed_csv_path, index=False)
    print(f"Saved processed CSV to: {processed_csv_path}")

    # Separate features and target
    X = merged.drop(columns=['CustomerId', 'is_high_risk'])
    y = merged['is_high_risk']
    print(f"Features shape: {X.shape}")

    # Build and fit preprocessor
    print("Building preprocessor...")
    preprocessor = build_preprocessor()

    # Find missing columns
    print("\nVerifying columns...")
    numeric_features = preprocessor.transformers[0][2]
    categorical_features = preprocessor.transformers[1][2]

    missing_num = [col for col in numeric_features if col not in X.columns]
    missing_cat = [col for col in categorical_features if col not in X.columns]

    if missing_num:
        print(f"WARNING: Missing numeric columns: {missing_num}")
        print("Filling with zeros...")
        for col in missing_num:
            X[col] = 0

    if missing_cat:
        print(f"WARNING: Missing categorical columns: {missing_cat}")
        print("Filling with 'missing'...")
        for col in missing_cat:
            X[col] = 'missing'

    # Fit preprocessor
    print("Fitting preprocessor...")
    X_processed = preprocessor.fit_transform(X)
    print(f"Processed features shape: {X_processed.shape}")

    # Save preprocessor
    preprocessor_path = models_dir / 'preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to: {preprocessor_path}")

    # Save processed arrays for training
    X_processed_path = processed_dir / 'X_processed.joblib'
    y_path = processed_dir / 'y.joblib'

    joblib.dump(X_processed, X_processed_path)
    joblib.dump(y, y_path)

    print(f"Saved X_processed to: {X_processed_path}")
    print(f"Saved y to: {y_path}")

    # Return processed data
    return X_processed, y, merged


if __name__ == "__main__":
    try:
        X, y, df = run_pipeline()
        print("\nPipeline completed successfully!")
        print(f"Final processed shape: {X.shape}")
        print(
            f"Target distribution:\n{
                pd.Series(y).value_counts(
                    normalize=True)}")
    except Exception as e:
        print(f"\nERROR in data processing pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
