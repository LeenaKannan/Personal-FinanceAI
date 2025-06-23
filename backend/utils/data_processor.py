# backend/utils/data_processor.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Comprehensive data processing utilities for Personal Finance AI.
    Handles Indian financial data with city-specific adjustments, inflation calculations,
    transaction preprocessing, ML feature engineering, and advanced analytics.
    """
    
    def __init__(self, inflation_rate: float = 0.06, city_cost_index: Optional[Dict[str, float]] = None):
        """
        Initialize DataProcessor with Indian market parameters.
        
        Args:
            inflation_rate: Annual inflation rate (default: 6% for India)
            city_cost_index: Cost of living index for Indian cities (Mumbai = 100 baseline)
        """
        self.inflation_rate = inflation_rate
        self.city_cost_index = city_cost_index or self._get_default_city_index()
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        # Indian financial year (April to March)
        self.financial_year_start_month = 4
        
        # Transaction categories for Indian users
        self.expense_categories = [
            'Housing', 'Food & Groceries', 'Transport', 'Utilities', 'Entertainment',
            'Self Care', 'Healthcare', 'Education', 'Shopping', 'Investment',
            'Insurance', 'EMI', 'Income', 'Transfer', 'Other'
        ]
        
        # Indian currency formatting
        self.currency_symbol = '₹'
        
    def _get_default_city_index(self) -> Dict[str, float]:
        """Get default cost of living index for major Indian cities."""
        return {
            'Mumbai': 100.0,      # Baseline
            'Delhi': 95.0,        # 5% lower than Mumbai
            'Bangalore': 88.0,    # 12% lower
            'Chennai': 82.0,      # 18% lower
            'Pune': 78.0,         # 22% lower
            'Hyderabad': 73.0,    # 27% lower
            'Kolkata': 75.0,      # 25% lower
            'Ahmedabad': 70.0,    # 30% lower
            'Jaipur': 65.0,       # 35% lower
            'Lucknow': 60.0,      # 40% lower
            'Surat': 68.0,        # 32% lower
            'Kanpur': 58.0,       # 42% lower
            'Nagpur': 62.0,       # 38% lower
            'Indore': 64.0,       # 36% lower
            'Thane': 95.0,        # Similar to Mumbai
            'Bhopal': 61.0,       # 39% lower
            'Visakhapatnam': 59.0, # 41% lower
            'Patna': 55.0,        # 45% lower
            'Vadodara': 67.0,     # 33% lower
            'Ghaziabad': 72.0,    # 28% lower
        }

    # === INFLATION AND COST ADJUSTMENTS ===
    
    def adjust_for_inflation(self, amount: Union[float, int, Decimal], years: float = 1) -> float:
        """
        Adjust amount for inflation over given years.
        
        Args:
            amount: Original amount
            years: Number of years to adjust for (can be fractional)
            
        Returns:
            float: Inflation-adjusted amount
        """
        try:
            if isinstance(amount, Decimal):
                amount = float(amount)
            return float(amount) * ((1 + self.inflation_rate) ** years)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid amount for inflation adjustment: {amount}. Error: {e}")
            return 0.0

    def adjust_for_city_cost(self, amount: Union[float, int], city: str) -> float:
        """
        Adjust amount based on city cost of living index.
        
        Args:
            amount: Original amount
            city: City name
            
        Returns:
            float: City cost-adjusted amount
        """
        try:
            base_index = self.city_cost_index.get('Mumbai', 100.0)
            city_index = self.city_cost_index.get(city.title(), base_index)
            return float(amount) * (city_index / base_index)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid amount for city cost adjustment: {amount}. Error: {e}")
            return 0.0

    def calculate_real_value(self, amount: float, city: str, years_ago: float = 0) -> float:
        """
        Calculate real value considering both inflation and city cost.
        
        Args:
            amount: Original amount
            city: City name
            years_ago: How many years ago this amount was relevant
            
        Returns:
            float: Real value in today's Mumbai equivalent
        """
        # First adjust for inflation to present value
        present_value = self.adjust_for_inflation(amount, years_ago)
        # Then adjust for city cost to Mumbai equivalent
        return self.adjust_for_city_cost(present_value, city)

    # === TRANSACTION PREPROCESSING ===
    
    def preprocess_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive transaction preprocessing for Indian financial data.
        
        Args:
            transactions_df: Raw transactions DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed transactions
        """
        if transactions_df.empty:
            logger.warning("Empty transactions DataFrame provided")
            return transactions_df
        
        df = transactions_df.copy()
        
        try:
            # Date processing
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])  # Remove invalid dates
                
                # Add derived date features
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day_of_week'] = df['date'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6])
                
                # Indian financial year
                df['financial_year'] = df['date'].apply(self._get_financial_year)
                df['quarter'] = df['date'].apply(self._get_financial_quarter)
            
            # Amount processing
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                df['amount_abs'] = df['amount'].abs()
                df['is_debit'] = df['amount'] < 0
                df['is_credit'] = df['amount'] > 0
                
                # Remove zero amounts
                df = df[df['amount'] != 0]
            
            # Description cleaning
            if 'description' in df.columns:
                df['description'] = df['description'].astype(str)
                df['description_clean'] = df['description'].apply(self._clean_description)
                df['description_length'] = df['description_clean'].str.len()
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Preprocessed {len(df)} transactions")
            return df
            
        except Exception as e:
            logger.error(f"Transaction preprocessing failed: {str(e)}")
            raise

    def _clean_description(self, description: str) -> str:
        """Clean transaction description."""
        if pd.isna(description):
            return "Unknown Transaction"
        
        # Convert to string and clean
        desc = str(description).strip().upper()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['UPI-', 'NEFT-', 'IMPS-', 'RTGS-', 'ATM-']
        for prefix in prefixes_to_remove:
            if desc.startswith(prefix):
                desc = desc[len(prefix):]
        
        # Remove extra spaces
        desc = ' '.join(desc.split())
        
        return desc[:100]  # Limit length

    def _get_financial_year(self, date: datetime) -> str:
        """Get Indian financial year (Apr-Mar) for a given date."""
        if date.month >= self.financial_year_start_month:
            return f"FY{date.year}-{date.year + 1}"
        else:
            return f"FY{date.year - 1}-{date.year}"

    def _get_financial_quarter(self, date: datetime) -> str:
        """Get financial quarter for Indian financial year."""
        if date.month >= 4 and date.month <= 6:
            return "Q1"
        elif date.month >= 7 and date.month <= 9:
            return "Q2"
        elif date.month >= 10 and date.month <= 12:
            return "Q3"
        else:
            return "Q4"

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in transaction data."""
        # Fill missing categories
        if 'category' in df.columns:
            df['category'] = df['category'].fillna('Other')
        
        # Fill missing descriptions
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('Unknown Transaction')
        
        # Fill missing amounts with 0 (though these should be removed)
        if 'amount' in df.columns:
            df['amount'] = df['amount'].fillna(0)
        
        return df

    # === EXPENSE CATEGORIZATION ===
    
    def categorize_expenses(self, transactions_df: pd.DataFrame, categorizer: Callable[[str], str]) -> pd.DataFrame:
        """
        Categorize transactions using a categorizer function.
        
        Args:
            transactions_df: Transactions DataFrame
            categorizer: Function that takes description and returns category
            
        Returns:
            pd.DataFrame: Transactions with category column
        """
        df = transactions_df.copy()
        
        try:
            if 'description' in df.columns:
                df['category'] = df['description'].apply(categorizer)
                
                # Validate categories
                df['category'] = df['category'].apply(
                    lambda x: x if x in self.expense_categories else 'Other'
                )
            else:
                logger.warning("No description column found for categorization")
                df['category'] = 'Other'
            
            return df
            
        except Exception as e:
            logger.error(f"Expense categorization failed: {str(e)}")
            df['category'] = 'Other'
            return df

    def auto_categorize_indian_transactions(self, description: str) -> str:
        """
        Auto-categorize Indian transaction descriptions.
        
        Args:
            description: Transaction description
            
        Returns:
            str: Predicted category
        """
        desc = description.upper()
        
        # Housing
        if any(keyword in desc for keyword in ['RENT', 'HOUSING', 'MAINTENANCE', 'SOCIETY']):
            return 'Housing'
        
        # Food & Groceries
        if any(keyword in desc for keyword in ['ZOMATO', 'SWIGGY', 'GROFER', 'BIGBASKET', 'RESTAURANT', 'FOOD', 'GROCERY']):
            return 'Food & Groceries'
        
        # Transport
        if any(keyword in desc for keyword in ['OLA', 'UBER', 'METRO', 'BUS', 'PETROL', 'DIESEL', 'FUEL']):
            return 'Transport'
        
        # Utilities
        if any(keyword in desc for keyword in ['ELECTRICITY', 'WATER', 'GAS', 'INTERNET', 'MOBILE', 'POSTPAID']):
            return 'Utilities'
        
        # Entertainment
        if any(keyword in desc for keyword in ['NETFLIX', 'AMAZON PRIME', 'HOTSTAR', 'MOVIE', 'BOOKMYSHOW']):
            return 'Entertainment'
        
        # Healthcare
        if any(keyword in desc for keyword in ['HOSPITAL', 'PHARMACY', 'DOCTOR', 'MEDICAL', 'HEALTH']):
            return 'Healthcare'
        
        # Shopping
        if any(keyword in desc for keyword in ['AMAZON', 'FLIPKART', 'MYNTRA', 'SHOPPING', 'MALL']):
            return 'Shopping'
        
        # Investment
        if any(keyword in desc for keyword in ['MUTUAL FUND', 'SIP', 'STOCK', 'ZERODHA', 'GROWW']):
            return 'Investment'
        
        # Income
        if any(keyword in desc for keyword in ['SALARY', 'CREDIT', 'REFUND', 'CASHBACK']):
            return 'Income'
        
        return 'Other'

    # === AGGREGATION AND ANALYTICS ===
    
    def aggregate_expenses(self, transactions_df: pd.DataFrame, group_by: Union[str, List[str]] = 'category') -> Dict[str, float]:
        """
        Aggregate expenses by specified grouping.
        
        Args:
            transactions_df: Transactions DataFrame
            group_by: Column(s) to group by
            
        Returns:
            dict: Aggregated expenses
        """
        try:
            df = transactions_df.copy()
            
            # Filter only expenses (negative amounts)
            expense_df = df[df['amount'] < 0].copy()
            expense_df['amount'] = expense_df['amount'].abs()  # Convert to positive
            
            if expense_df.empty:
                return {}
            
            if isinstance(group_by, str):
                return expense_df.groupby(group_by)['amount'].sum().to_dict()
            else:
                return expense_df.groupby(group_by)['amount'].sum().to_dict()
                
        except Exception as e:
            logger.error(f"Expense aggregation failed: {str(e)}")
            return {}

    def calculate_monthly_summary(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive monthly financial summary.
        
        Args:
            transactions_df: Transactions DataFrame
            
        Returns:
            dict: Monthly summary statistics
        """
        try:
            df = transactions_df.copy()
            
            if df.empty or 'date' not in df.columns:
                return {}
            
            # Current month data
            current_month = datetime.now().replace(day=1)
            monthly_data = df[df['date'] >= current_month]
            
            # Calculate metrics
            total_income = monthly_data[monthly_data['amount'] > 0]['amount'].sum()
            total_expenses = abs(monthly_data[monthly_data['amount'] < 0]['amount'].sum())
            net_savings = total_income - total_expenses
            savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
            
            # Category breakdown
            category_expenses = self.aggregate_expenses(monthly_data)
            
            # Transaction counts
            total_transactions = len(monthly_data)
            expense_transactions = len(monthly_data[monthly_data['amount'] < 0])
            income_transactions = len(monthly_data[monthly_data['amount'] > 0])
            
            return {
                'period': current_month.strftime('%Y-%m'),
                'total_income': round(total_income, 2),
                'total_expenses': round(total_expenses, 2),
                'net_savings': round(net_savings, 2),
                'savings_rate': round(savings_rate, 2),
                'category_expenses': category_expenses,
                'transaction_counts': {
                    'total': total_transactions,
                    'expenses': expense_transactions,
                    'income': income_transactions
                },
                'average_transaction_size': round(total_expenses / expense_transactions, 2) if expense_transactions > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Monthly summary calculation failed: {str(e)}")
            return {}

    # === ML FEATURE ENGINEERING ===
    
    def prepare_ml_features(self, user_profile: Dict[str, Any], transactions_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prepare features for ML model prediction.
        
        Args:
            user_profile: User profile dictionary
            transactions_df: Optional historical transactions
            
        Returns:
            np.ndarray: Feature array for ML model
        """
        try:
            features = []
            
            # Basic user features
            features.extend([
                user_profile.get('income', 0),
                self.city_cost_index.get(user_profile.get('city', 'Mumbai'), 100),
                1 if user_profile.get('gender', 'male').lower() == 'male' else 0,
                user_profile.get('age', 30)
            ])
            
            # Historical spending patterns (if available)
            if transactions_df is not None and not transactions_df.empty:
                monthly_summary = self.calculate_monthly_summary(transactions_df)
                features.extend([
                    monthly_summary.get('total_expenses', 0),
                    monthly_summary.get('savings_rate', 0),
                    monthly_summary.get('average_transaction_size', 0)
                ])
            else:
                features.extend([0, 0, 0])  # Default values
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"ML feature preparation failed: {str(e)}")
            # Return default features
            return np.array([50000, 100, 1, 30, 0, 0, 0]).reshape(1, -1)

    def predict_expenses(self, user_profile: Dict[str, Any], model: Any, transactions_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Predict expenses using ML model with enhanced features.
        
        Args:
            user_profile: User profile dictionary
            model: Trained ML model
            transactions_df: Optional historical transactions
            
        Returns:
            dict: Predicted expenses by category
        """
        try:
            features = self.prepare_ml_features(user_profile, transactions_df)
            
            # Get prediction from model
            if hasattr(model, 'predict'):
                prediction = model.predict(features)
                
                # Convert to category-wise predictions
                if isinstance(prediction, np.ndarray):
                    if prediction.ndim > 1:
                        prediction = prediction[0]
                    
                    # Map to categories (assuming model outputs in specific order)
                    categories = ['Housing', 'Food & Groceries', 'Transport', 'Utilities', 'Entertainment', 'Self Care', 'Other']
                    result = {}
                    
                    for i, category in enumerate(categories):
                        if i < len(prediction):
                            result[category] = round(float(prediction[i]), 2)
                        else:
                            result[category] = 0.0
                    
                    return result
                else:
                    return prediction
            else:
                logger.warning("Model does not have predict method")
                return self._get_default_prediction(user_profile)
                
        except Exception as e:
            logger.error(f"Expense prediction failed: {str(e)}")
            return self._get_default_prediction(user_profile)

    def _get_default_prediction(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Get default expense prediction based on user profile."""
        income = user_profile.get('income', 50000)
        city = user_profile.get('city', 'Mumbai')
        
        # Adjust for city cost
        base_expenses = income * 0.7  # Assume 70% of income as expenses
        adjusted_expenses = self.adjust_for_city_cost(base_expenses, city)
        
        # Default distribution
        return {
            'Housing': round(adjusted_expenses * 0.35, 2),
            'Food & Groceries': round(adjusted_expenses * 0.20, 2),
            'Transport': round(adjusted_expenses * 0.12, 2),
            'Utilities': round(adjusted_expenses * 0.08, 2),
            'Entertainment': round(adjusted_expenses * 0.07, 2),
            'Self Care': round(adjusted_expenses * 0.05, 2),
            'Other': round(adjusted_expenses * 0.13, 2)
        }

    # === UTILITY METHODS ===
    
    def format_currency(self, amount: float) -> str:
        """Format amount as Indian currency."""
        return f"{self.currency_symbol}{amount:,.2f}"

    def calculate_expense_trends(self, transactions_df: pd.DataFrame, periods: int = 6) -> Dict[str, List[float]]:
        """
        Calculate expense trends over last N periods.
        
        Args:
            transactions_df: Transactions DataFrame
            periods: Number of months to analyze
            
        Returns:
            dict: Trends by category
        """
        try:
            df = transactions_df.copy()
            
            if df.empty or 'date' not in df.columns:
                return {}
            
            # Filter last N months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=periods * 30)
            df = df[df['date'] >= start_date]
            
            # Group by month and category
            df['month'] = df['date'].dt.to_period('M')
            expense_df = df[df['amount'] < 0].copy()
            expense_df['amount'] = expense_df['amount'].abs()
            
            trends = {}
            for category in self.expense_categories:
                category_data = expense_df[expense_df['category'] == category]
                monthly_amounts = category_data.groupby('month')['amount'].sum()
                trends[category] = monthly_amounts.values.tolist()
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {str(e)}")
            return {}

    def save_processor_state(self, file_path: str) -> None:
        """Save processor state (scalers, encoders, etc.) to file."""
        try:
            state = {
                'inflation_rate': self.inflation_rate,
                'city_cost_index': self.city_cost_index,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'imputers': self.imputers
            }
            joblib.dump(state, file_path)
            logger.info(f"Processor state saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save processor state: {str(e)}")

    def load_processor_state(self, file_path: str) -> None:
        """Load processor state from file."""
        try:
            if os.path.exists(file_path):
                state = joblib.load(file_path)
                self.inflation_rate = state.get('inflation_rate', 0.06)
                self.city_cost_index = state.get('city_cost_index', self._get_default_city_index())
                self.scalers = state.get('scalers', {})
                self.encoders = state.get('encoders', {})
                self.imputers = state.get('imputers', {})
                logger.info(f"Processor state loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load processor state: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    # Test the data processor
    print("Testing DataProcessor...")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Test data
    sample_data = {
        'date': ['2025-06-01', '2025-06-02', '2025-06-03', '2025-06-04'],
        'description': ['ZOMATO ONLINE PAYMENT', 'RENT TRANSFER', 'SALARY CREDIT', 'OLA CABS'],
        'amount': [-850, -28000, 80000, -1200]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Preprocess transactions
    df_processed = processor.preprocess_transactions(df)
    print(f"Processed {len(df_processed)} transactions")
    
    # Auto-categorize
    df_categorized = processor.categorize_expenses(
        df_processed, 
        processor.auto_categorize_indian_transactions
    )
    print("Categories assigned:", df_categorized['category'].tolist())
    
    # Aggregate expenses
    aggregated = processor.aggregate_expenses(df_categorized)
    print("Aggregated expenses:", aggregated)
    
    # Monthly summary
    summary = processor.calculate_monthly_summary(df_categorized)
    print("Monthly summary:", summary)
    
    # Test predictions
    user_profile = {
        'income': 80000,
        'city': 'Mumbai',
        'gender': 'male',
        'age': 28
    }
    
    default_prediction = processor._get_default_prediction(user_profile)
    print("Default prediction:", default_prediction)
    
    print("All tests completed successfully!")
