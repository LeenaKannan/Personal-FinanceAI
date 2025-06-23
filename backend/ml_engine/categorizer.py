# Transaction categorization
# backend/ml_engine/categorizer.py

import re
from typing import List

class TransactionCategorizer:
    """
    Categorizes transaction descriptions using rule-based and ML/NLP models.
    """

    def __init__(self, model_path: str = None):
        # Placeholder for ML/NLP model (e.g., BERT, spaCy)
        self.model = None
        if model_path:
            # Load your ML model here
            pass

        # Simple keyword-based rules
        self.rules = [
            (re.compile(r"zomato|swiggy|restaurant|food|grocer", re.I), "Food & Groceries"),
            (re.compile(r"rent|lease", re.I), "Housing"),
            (re.compile(r"ola|uber|cab|auto|bus|metro|transport", re.I), "Transport"),
            (re.compile(r"electricity|water|gas|utility|bill", re.I), "Utilities"),
            (re.compile(r"movie|netflix|prime|entertainment|bookmyshow", re.I), "Entertainment"),
            (re.compile(r"salon|spa|self care|parlor", re.I), "Self Care"),
            (re.compile(r"salary|credit|income", re.I), "Income"),
        ]

    def categorize(self, description: str) -> str:
        # Rule-based categorization
        for pattern, category in self.rules:
            if pattern.search(description):
                return category
        # Fallback
        return "Other"

    def batch_categorize(self, descriptions: List[str]) -> List[str]:
        return [self.categorize(desc) for desc in descriptions]
