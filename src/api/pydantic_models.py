from pydantic import BaseModel
from typing import List, Dict, Optional


class Transaction(BaseModel):
    TransactionId: str
    AccountId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: Optional[str] = None
    FraudResult: Optional[int] = None


class PredictionRequest(BaseModel):
    transactions: List[Transaction]


class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
    risk_category: str
    credit_score: int
    recommended_limit: float
