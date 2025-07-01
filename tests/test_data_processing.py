import pytest
import pandas as pd
import numpy as np
from src.data_processing import RFMTransformer, create_proxy_target, FeatureAggregator

def test_rfm_transformer():
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C2'],
        'TransactionStartTime': [
            '2023-01-01 10:00:00',
            '2023-01-03 10:00:00',
            '2023-01-02 10:00:00',
            '2023-01-05 10:00:00',
            '2023-01-07 10:00:00'
        ],
        'Amount': [100.0, 50.0, 200.0, 150.0, 300.0]
    }
    df = pd.DataFrame(data)
    snapshot_date = pd.Timestamp('2023-01-10')
    transformer = RFMTransformer(snapshot_date)
    rfm = transformer.transform(df)
    
    # Check results
    assert rfm.shape == (2, 4)
    assert rfm[rfm['CustomerId'] == 'C1']['Recency'].iloc[0] == 7
    assert rfm[rfm['CustomerId'] == 'C1']['Frequency'].iloc[0] == 2
    assert rfm[rfm['CustomerId'] == 'C1']['Monetary'].iloc[0] == 150.0
    assert rfm[rfm['CustomerId'] == 'C2']['Recency'].iloc[0] == 3
    assert rfm[rfm['CustomerId'] == 'C2']['Frequency'].iloc[0] == 3
    assert rfm[rfm['CustomerId'] == 'C2']['Monetary'].iloc[0] == 650.0

def test_create_proxy_target():
    data = {
        'CustomerId': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'Recency': [100, 50, 10, 200, 150],
        'Frequency': [5, 10, 20, 2, 3],
        'Monetary': [500, 1000, 2000, 100, 200]
    }
    rfm_df = pd.DataFrame(data)
    result = create_proxy_target(rfm_df)
    
    # Check high-risk assignment
    assert 'is_high_risk' in result.columns
    assert result[result['CustomerId'] == 'C4']['is_high_risk'].iloc[0] == 1
    assert result[result['CustomerId'] == 'C3']['is_high_risk'].iloc[0] == 0
    assert result['is_high_risk'].mean() > 0.1  # At least some high-risk

def test_feature_aggregator():
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C2'],
        'Amount': [100, 200, 50, 150, 300],
        'Value': [100, 200, 50, 150, 300],
        'ProductCategory': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Electronics'],
        'ChannelId': ['Web', 'App', 'App', 'Web', 'App'],
        'TransactionStartTime': [
            '2023-01-01 08:00:00',
            '2023-01-02 14:00:00',
            '2023-01-01 10:00:00',
            '2023-01-03 16:00:00',
            '2023-01-04 20:00:00'
        ]
    }
    df = pd.DataFrame(data)
    aggregator = FeatureAggregator()
    result = aggregator.transform(df)
    
    # Check results
    assert result.shape == (2, 14)
    assert result[result['CustomerId'] == 'C1']['Amount_sum'].iloc[0] == 300
    assert result[result['CustomerId'] == 'C2']['ProductCategory_mode'].iloc[0] == 'Electronics'
    assert result[result['CustomerId'] == 'C2']['ChannelId_mode'].iloc[0] == 'App'
    assert result[result['CustomerId'] == 'C1']['TransactionHour_mean'].iloc[0] == 11.0