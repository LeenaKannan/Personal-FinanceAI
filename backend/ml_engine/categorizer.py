# Transaction categorization with Indian context
# backend/ml_engine/categorizer.py

import re
import logging
import joblib
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import asyncio
import threading
from functools import lru_cache
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CategoryRule:
    """Data class for categorization rules"""
    pattern: re.Pattern
    category: str
    confidence: float
    subcategory: Optional[str] = None

class TransactionCategorizer:
    """
    Advanced transaction categorizer for Indian financial ecosystem.
    Supports both rule-based and ML-based categorization with fallback logic.
    """

    def __init__(self, model_path: Optional[str] = None, enable_caching: bool = True):
        """
        Initialize the categorizer with optional ML model and caching.
        
        Args:
            model_path: Path to trained ML model (pickle/joblib format)
            enable_caching: Whether to enable LRU caching for repeated categorizations
        """
        self.model = None
        self.model_path = model_path
        self.enable_caching = enable_caching
        self._lock = threading.Lock()
        
        # Load ML model if available
        if model_path:
            self._load_model()
        
        # Indian-specific categorization rules with confidence scores
        self.rules = self._initialize_rules()
        
        # Category mapping for Indian financial ecosystem
        self.category_hierarchy = self._initialize_category_hierarchy()
        
        # UPI and Indian payment gateway patterns
        self.upi_patterns = self._initialize_upi_patterns()
        
        logger.info(f"TransactionCategorizer initialized with {len(self.rules)} rules")

    def _load_model(self) -> None:
        """Load the ML model with error handling"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"ML model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}. Using rule-based fallback.")
            self.model = None

    def _initialize_rules(self) -> List[CategoryRule]:
        """Initialize comprehensive Indian context-aware categorization rules"""
        return [
            # Food & Groceries (Indian specific)
            CategoryRule(re.compile(r"zomato|swiggy|uber eats|dominos|pizza hut|kfc", re.I), 
                        "Food & Groceries", 0.95, "Online Food"),
            CategoryRule(re.compile(r"big bazaar|dmart|reliance fresh|more|spencer|grocery", re.I), 
                        "Food & Groceries", 0.9, "Grocery Shopping"),
            CategoryRule(re.compile(r"restaurant|cafe|hotel|dhaba|mess|canteen", re.I), 
                        "Food & Groceries", 0.85, "Dining Out"),
            CategoryRule(re.compile(r"milk|bread|vegetables|fruits|market|sabzi", re.I), 
                        "Food & Groceries", 0.8, "Daily Essentials"),
            
            # Housing & Utilities
            CategoryRule(re.compile(r"rent|lease|maintenance|society|apartment", re.I), 
                        "Housing", 0.95, "Rent & Maintenance"),
            CategoryRule(re.compile(r"electricity|electric|power|mseb|bescom|kseb", re.I), 
                        "Utilities", 0.9, "Electricity"),
            CategoryRule(re.compile(r"gas|lpg|indane|bharat gas|hp gas", re.I), 
                        "Utilities", 0.9, "Gas"),
            CategoryRule(re.compile(r"water|municipal|corporation|tax|property", re.I), 
                        "Utilities", 0.85, "Water & Tax"),
            CategoryRule(re.compile(r"broadband|internet|wifi|airtel|jio|bsnl", re.I), 
                        "Utilities", 0.85, "Internet"),
            
            # Transportation (Indian specific)
            CategoryRule(re.compile(r"ola|uber|rapido|auto|rickshaw", re.I), 
                        "Transport", 0.95, "Ride Sharing"),
            CategoryRule(re.compile(r"metro|local|train|railway|irctc", re.I), 
                        "Transport", 0.9, "Public Transport"),
            CategoryRule(re.compile(r"petrol|diesel|fuel|pump|hp|bharat petroleum|ioc", re.I), 
                        "Transport", 0.9, "Fuel"),
            CategoryRule(re.compile(r"parking|toll|highway|fastag", re.I), 
                        "Transport", 0.85, "Vehicle Expenses"),
            CategoryRule(re.compile(r"bus|volvo|state transport|ksrtc|msrtc", re.I), 
                        "Transport", 0.8, "Bus Travel"),
            
            # Entertainment & Lifestyle
            CategoryRule(re.compile(r"netflix|prime|hotstar|zee5|sonyliv|voot", re.I), 
                        "Entertainment", 0.9, "Streaming"),
            CategoryRule(re.compile(r"bookmyshow|pvr|inox|cinema|movie|theatre", re.I), 
                        "Entertainment", 0.9, "Movies"),
            CategoryRule(re.compile(r"spotify|gaana|wynk|music|jio saavn", re.I), 
                        "Entertainment", 0.85, "Music"),
            CategoryRule(re.compile(r"gym|fitness|yoga|sports|swimming", re.I), 
                        "Health & Fitness", 0.85, "Fitness"),
            
            # Shopping (Indian e-commerce)
            CategoryRule(re.compile(r"amazon|flipkart|myntra|ajio|nykaa|snapdeal", re.I), 
                        "Shopping", 0.9, "Online Shopping"),
            CategoryRule(re.compile(r"mall|shop|store|market|bazaar|clothing", re.I), 
                        "Shopping", 0.8, "Retail Shopping"),
            
            # Healthcare
            CategoryRule(re.compile(r"hospital|clinic|doctor|medical|pharmacy|medicine", re.I), 
                        "Healthcare", 0.9, "Medical"),
            CategoryRule(re.compile(r"apollo|max|fortis|manipal|insurance|mediclaim", re.I), 
                        "Healthcare", 0.85, "Healthcare Services"),
            
            # Financial Services
            CategoryRule(re.compile(r"sip|mutual fund|investment|stock|equity|trading", re.I), 
                        "Investment", 0.9, "Investments"),
            CategoryRule(re.compile(r"loan|emi|credit card|repayment|interest", re.I), 
                        "Loan & EMI", 0.9, "Debt Payment"),
            CategoryRule(re.compile(r"insurance|premium|lic|policy", re.I), 
                        "Insurance", 0.85, "Insurance Premium"),
            
            # Education
            CategoryRule(re.compile(r"school|college|university|fees|tuition|course", re.I), 
                        "Education", 0.9, "Education Fees"),
            CategoryRule(re.compile(r"book|stationery|study|exam|coaching", re.I), 
                        "Education", 0.8, "Educational Materials"),
            
            # Self Care & Personal
            CategoryRule(re.compile(r"salon|parlour|spa|beauty|grooming", re.I), 
                        "Self Care", 0.85, "Beauty & Grooming"),
            CategoryRule(re.compile(r"temple|donation|charity|religious", re.I), 
                        "Personal", 0.8, "Religious & Charity"),
            
            # Income patterns
            CategoryRule(re.compile(r"salary|wages|bonus|incentive|income|credit", re.I), 
                        "Income", 0.9, "Salary"),
            CategoryRule(re.compile(r"interest|dividend|returns|profit", re.I), 
                        "Income", 0.8, "Investment Income"),
        ]

    def _initialize_category_hierarchy(self) -> Dict[str, List[str]]:
        """Initialize category hierarchy for better organization"""
        return {
            "Food & Groceries": ["Online Food", "Grocery Shopping", "Dining Out", "Daily Essentials"],
            "Housing": ["Rent & Maintenance", "Home Utilities"],
            "Utilities": ["Electricity", "Gas", "Water & Tax", "Internet", "Phone"],
            "Transport": ["Ride Sharing", "Public Transport", "Fuel", "Vehicle Expenses", "Bus Travel"],
            "Entertainment": ["Streaming", "Movies", "Music", "Gaming", "Events"],
            "Shopping": ["Online Shopping", "Retail Shopping", "Clothing", "Electronics"],
            "Healthcare": ["Medical", "Healthcare Services", "Pharmacy", "Fitness"],
            "Investment": ["Investments", "Savings", "Fixed Deposits"],
            "Loan & EMI": ["Debt Payment", "Credit Card", "Personal Loan"],
            "Insurance": ["Insurance Premium", "Health Insurance", "Life Insurance"],
            "Education": ["Education Fees", "Educational Materials", "Online Courses"],
            "Self Care": ["Beauty & Grooming", "Personal Care"],
            "Personal": ["Religious & Charity", "Gifts", "Miscellaneous"],
            "Income": ["Salary", "Investment Income", "Business Income"],
            "Other": ["Uncategorized", "Cash Withdrawal"]
        }

    def _initialize_upi_patterns(self) -> Dict[str, str]:
        """Initialize UPI and payment gateway patterns"""
        return {
            "paytm": "Digital Payments",
            "phonepe": "Digital Payments",
            "googlepay": "Digital Payments",
            "bhim": "Digital Payments",
            "upi": "Digital Payments",
            "neft": "Bank Transfer",
            "rtgs": "Bank Transfer",
            "imps": "Bank Transfer",
            "razorpay": "Online Payment",
            "billdesk": "Bill Payment",
            "ccavenue": "Online Payment"
        }

    @lru_cache(maxsize=1000)
    def _cached_categorize(self, description: str) -> Tuple[str, str, float]:
        """Cached version of categorization for performance"""
        return self._categorize_internal(description)

    def _categorize_internal(self, description: str) -> Tuple[str, str, float]:
        """Internal categorization logic"""
        if not description or not description.strip():
            return "Other", "Uncategorized", 0.0
        
        description_clean = description.strip().lower()
        
        # Check UPI patterns first
        for pattern, category in self.upi_patterns.items():
            if pattern in description_clean:
                return "Digital Payments", category, 0.8
        
        # Rule-based categorization with confidence scoring
        best_match = None
        highest_confidence = 0.0
        
        for rule in self.rules:
            if rule.pattern.search(description):
                if rule.confidence > highest_confidence:
                    highest_confidence = rule.confidence
                    best_match = rule
        
        if best_match:
            return best_match.category, best_match.subcategory or "General", highest_confidence
        
        # ML model prediction if available
        if self.model:
            try:
                category, confidence = self._ml_predict(description)
                if confidence > 0.5:  # Threshold for ML predictions
                    return category, "ML Predicted", confidence
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # Fallback to Other category
        return "Other", "Uncategorized", 0.1

    def _ml_predict(self, description: str) -> Tuple[str, float]:
        """ML model prediction with error handling"""
        # This is a placeholder for actual ML model prediction
        # In production, you would vectorize the text and predict
        try:
            # Assuming the model expects vectorized input
            # features = self.vectorizer.transform([description])
            # prediction = self.model.predict_proba(features)[0]
            # category_idx = np.argmax(prediction)
            # confidence = prediction[category_idx]
            # return self.categories[category_idx], float(confidence)
            
            # Placeholder implementation
            return "Other", 0.5
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return "Other", 0.0

    def categorize(self, description: str) -> Dict[str, any]:
        """
        Categorize a single transaction description.
        
        Args:
            description: Transaction description text
            
        Returns:
            Dict containing category, subcategory, confidence, and metadata
        """
        try:
            if self.enable_caching:
                category, subcategory, confidence = self._cached_categorize(description)
            else:
                category, subcategory, confidence = self._categorize_internal(description)
            
            return {
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
                "method": "ml" if confidence > 0.5 and self.model else "rule",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Categorization error for '{description}': {e}")
            return {
                "category": "Other",
                "subcategory": "Error",
                "confidence": 0.0,
                "method": "fallback",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def batch_categorize(self, descriptions: List[str]) -> List[Dict[str, any]]:
        """
        Categorize multiple transaction descriptions efficiently.
        
        Args:
            descriptions: List of transaction descriptions
            
        Returns:
            List of categorization results
        """
        results = []
        try:
            for desc in descriptions:
                result = self.categorize(desc)
                results.append(result)
            
            logger.info(f"Batch categorized {len(descriptions)} transactions")
            return results
            
        except Exception as e:
            logger.error(f"Batch categorization error: {e}")
            # Return partial results with error indicators
            while len(results) < len(descriptions):
                results.append({
                    "category": "Other",
                    "subcategory": "Error",
                    "confidence": 0.0,
                    "method": "fallback",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
            return results

    async def async_batch_categorize(self, descriptions: List[str], batch_size: int = 100) -> List[Dict[str, any]]:
        """
        Asynchronous batch categorization for large datasets.
        
        Args:
            descriptions: List of transaction descriptions
            batch_size: Number of items to process in each batch
            
        Returns:
            List of categorization results
        """
        results = []
        
        try:
            # Process in batches to avoid memory issues
            for i in range(0, len(descriptions), batch_size):
                batch = descriptions[i:i + batch_size]
                
                # Run categorization in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    None, self.batch_categorize, batch
                )
                
                results.extend(batch_results)
                
                # Optional: Add small delay to prevent overwhelming the system
                if len(descriptions) > 1000:
                    await asyncio.sleep(0.01)
            
            logger.info(f"Async batch categorized {len(descriptions)} transactions")
            return results
            
        except Exception as e:
            logger.error(f"Async batch categorization error: {e}")
            return results

    def get_category_stats(self, transactions: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Generate statistics about categorized transactions.
        
        Args:
            transactions: List of categorized transactions
            
        Returns:
            Statistics dictionary
        """
        try:
            total_transactions = len(transactions)
            if total_transactions == 0:
                return {"error": "No transactions provided"}
            
            category_counts = {}
            confidence_scores = []
            method_counts = {"rule": 0, "ml": 0, "fallback": 0}
            
            for txn in transactions:
                # Count categories
                category = txn.get("category", "Other")
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Track confidence scores
                confidence = txn.get("confidence", 0.0)
                confidence_scores.append(confidence)
                
                # Count methods
                method = txn.get("method", "fallback")
                method_counts[method] = method_counts.get(method, 0) + 1
            
            # Calculate statistics
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                "total_transactions": total_transactions,
                "unique_categories": len(category_counts),
                "category_distribution": category_counts,
                "average_confidence": float(avg_confidence),
                "method_distribution": method_counts,
                "high_confidence_ratio": len([c for c in confidence_scores if c > 0.8]) / total_transactions,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating category stats: {e}")
            return {"error": str(e)}

    def update_rules(self, new_rules: List[Dict[str, any]]) -> bool:
        """
        Update categorization rules dynamically.
        
        Args:
            new_rules: List of rule dictionaries with pattern, category, confidence
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                for rule_dict in new_rules:
                    pattern = re.compile(rule_dict["pattern"], re.I)
                    rule = CategoryRule(
                        pattern=pattern,
                        category=rule_dict["category"],
                        confidence=rule_dict.get("confidence", 0.8),
                        subcategory=rule_dict.get("subcategory")
                    )
                    self.rules.append(rule)
                
                # Clear cache to ensure new rules are applied
                if self.enable_caching:
                    self._cached_categorize.cache_clear()
                
                logger.info(f"Added {len(new_rules)} new categorization rules")
                return True
                
        except Exception as e:
            logger.error(f"Error updating rules: {e}")
            return False

    def export_model_data(self) -> Dict[str, any]:
        """Export model data for training or analysis"""
        try:
            return {
                "rules_count": len(self.rules),
                "categories": list(self.category_hierarchy.keys()),
                "upi_patterns": self.upi_patterns,
                "model_loaded": self.model is not None,
                "cache_enabled": self.enable_caching,
                "cache_info": self._cached_categorize.cache_info() if self.enable_caching else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error exporting model data: {e}")
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Initialize categorizer
    categorizer = TransactionCategorizer()
    
    # Test transactions (Indian context)
    test_transactions = [
        "ZOMATO ORDER 12345",
        "UBER RIDE - BANGALORE",
        "ELECTRICITY BILL BESCOM",
        "AMAZON.IN PURCHASE",
        "SIP HDFC MUTUAL FUND",
        "SALARY CREDIT TCS",
        "UPI P2P PAYTM",
        "PETROL PUMP HP",
        "NETFLIX SUBSCRIPTION",
        "RENT PAYMENT APRIL 2024"
    ]
    
    # Single categorization
    print("Single categorization:")
    for desc in test_transactions[:3]:
        result = categorizer.categorize(desc)
        print(f"{desc}: {result['category']} ({result['confidence']:.2f})")
    
    # Batch categorization
    print("\nBatch categorization:")
    batch_results = categorizer.batch_categorize(test_transactions)
    for i, result in enumerate(batch_results):
        print(f"{test_transactions[i]}: {result['category']} - {result['subcategory']}")
    
    # Category statistics
    print("\nCategory Statistics:")
    stats = categorizer.get_category_stats(batch_results)
    print(f"Total: {stats['total_transactions']}")
    print(f"Categories: {stats['unique_categories']}")
    print(f"Avg Confidence: {stats['average_confidence']:.2f}")
