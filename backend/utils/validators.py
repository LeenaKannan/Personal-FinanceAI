# backend/utils/validators.py

import re
from datetime import datetime, date
from typing import Union, List, Optional, Any
from decimal import Decimal, InvalidOperation
import phonenumbers
from phonenumbers import NumberParseException

class Validators:
    """
    Comprehensive validation utilities for Personal Finance AI application.
    Includes validators for Indian financial data, user inputs, and system constraints.
    """
    
    # Indian cities supported by the application
    VALID_INDIAN_CITIES = [
        'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Hyderabad', 
        'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Kanpur',
        'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad',
        'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik',
        'Faridabad', 'Meerut', 'Rajkot', 'Kalyan-Dombivli', 'Vasai-Virar',
        'Varanasi', 'Srinagar', 'Aurangabad', 'Dhanbad', 'Amritsar',
        'Navi Mumbai', 'Allahabad', 'Ranchi', 'Howrah', 'Coimbatore',
        'Jabalpur', 'Gwalior', 'Vijayawada', 'Jodhpur', 'Madurai',
        'Raipur', 'Kota', 'Guwahati', 'Chandigarh', 'Solapur'
    ]
    
    # Valid transaction categories
    VALID_CATEGORIES = [
        'Housing', 'Food & Groceries', 'Transport', 'Utilities', 'Entertainment',
        'Self Care', 'Healthcare', 'Education', 'Shopping', 'Investment',
        'Insurance', 'EMI', 'Income', 'Transfer', 'Other'
    ]
    
    # Valid investment types
    VALID_INVESTMENT_TYPES = [
        'Mutual Fund', 'Stock', 'Fixed Deposit', 'PPF', 'ELSS', 'Bond',
        'Gold', 'Real Estate', 'Cryptocurrency', 'NSC', 'EPF', 'NPS'
    ]
    
    @staticmethod
    def validate_date(date_str: Union[str, date], date_format: str = '%Y-%m-%d') -> bool:
        """
        Validate date string or date object.
        
        Args:
            date_str: Date string or date object to validate
            date_format: Expected date format (default: '%Y-%m-%d')
            
        Returns:
            bool: True if valid date, False otherwise
        """
        if isinstance(date_str, date):
            return True
            
        if not isinstance(date_str, str):
            return False
            
        try:
            parsed_date = datetime.strptime(date_str, date_format)
            # Check if date is not in future (for transaction dates)
            return parsed_date.date() <= datetime.now().date()
        except ValueError:
            return False
    
    @staticmethod
    def validate_amount(amount: Union[str, int, float, Decimal]) -> bool:
        """
        Validate monetary amount.
        
        Args:
            amount: Amount to validate
            
        Returns:
            bool: True if valid amount (>= 0), False otherwise
        """
        try:
            if isinstance(amount, str):
                # Remove common currency symbols and spaces
                amount = amount.replace('₹', '').replace(',', '').strip()
            
            val = Decimal(str(amount))
            return val >= 0 and val <= Decimal('10000000')  # Max 1 crore
        except (ValueError, TypeError, InvalidOperation):
            return False
    
    @staticmethod
    def validate_city(city: str, valid_cities: Optional[List[str]] = None) -> bool:
        """
        Validate Indian city name.
        
        Args:
            city: City name to validate
            valid_cities: Optional list of valid cities (uses default if None)
            
        Returns:
            bool: True if valid city, False otherwise
        """
        if not isinstance(city, str):
            return False
            
        cities_list = valid_cities or Validators.VALID_INDIAN_CITIES
        return city.strip().title() in cities_list
    
    @staticmethod
    def validate_gender(gender: str) -> bool:
        """
        Validate gender input.
        
        Args:
            gender: Gender string to validate
            
        Returns:
            bool: True if valid gender, False otherwise
        """
        if not isinstance(gender, str):
            return False
        return gender.lower().strip() in ['male', 'female', 'other', 'prefer not to say']
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format.
        
        Args:
            email: Email string to validate
            
        Returns:
            bool: True if valid email format, False otherwise
        """
        if not isinstance(email, str):
            return False
            
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email.strip()) is not None
    
    @staticmethod
    def validate_phone(phone: str, country_code: str = 'IN') -> bool:
        """
        Validate Indian phone number.
        
        Args:
            phone: Phone number string to validate
            country_code: Country code (default: 'IN' for India)
            
        Returns:
            bool: True if valid phone number, False otherwise
        """
        try:
            parsed_number = phonenumbers.parse(phone, country_code)
            return phonenumbers.is_valid_number(parsed_number)
        except NumberParseException:
            return False
    
    @staticmethod
    def validate_pan(pan: str) -> bool:
        """
        Validate Indian PAN (Permanent Account Number) format.
        
        Args:
            pan: PAN string to validate
            
        Returns:
            bool: True if valid PAN format, False otherwise
        """
        if not isinstance(pan, str):
            return False
            
        pan = pan.upper().strip()
        pattern = r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$"
        return re.match(pattern, pan) is not None
    
    @staticmethod
    def validate_aadhar(aadhar: str) -> bool:
        """
        Validate Indian Aadhar number format.
        
        Args:
            aadhar: Aadhar string to validate
            
        Returns:
            bool: True if valid Aadhar format, False otherwise
        """
        if not isinstance(aadhar, str):
            return False
            
        # Remove spaces and hyphens
        aadhar = aadhar.replace(' ', '').replace('-', '').strip()
        
        # Check if 12 digits
        if not (len(aadhar) == 12 and aadhar.isdigit()):
            return False
            
        # Aadhar cannot start with 0 or 1
        return aadhar[0] not in ['0', '1']
    
    @staticmethod
    def validate_ifsc(ifsc: str) -> bool:
        """
        Validate Indian IFSC (Indian Financial System Code) format.
        
        Args:
            ifsc: IFSC string to validate
            
        Returns:
            bool: True if valid IFSC format, False otherwise
        """
        if not isinstance(ifsc, str):
            return False
            
        ifsc = ifsc.upper().strip()
        pattern = r"^[A-Z]{4}0[A-Z0-9]{6}$"
        return re.match(pattern, ifsc) is not None
    
    @staticmethod
    def validate_income(income: Union[str, int, float]) -> bool:
        """
        Validate monthly income amount.
        
        Args:
            income: Income amount to validate
            
        Returns:
            bool: True if valid income (between 10K and 50L per month), False otherwise
        """
        try:
            if isinstance(income, str):
                income = income.replace('₹', '').replace(',', '').strip()
            
            val = float(income)
            return 10000 <= val <= 5000000  # Between 10K and 50L per month
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_age(age: Union[str, int]) -> bool:
        """
        Validate user age.
        
        Args:
            age: Age to validate
            
        Returns:
            bool: True if valid age (18-100), False otherwise
        """
        try:
            age_val = int(age)
            return 18 <= age_val <= 100
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_category(category: str) -> bool:
        """
        Validate transaction category.
        
        Args:
            category: Category string to validate
            
        Returns:
            bool: True if valid category, False otherwise
        """
        if not isinstance(category, str):
            return False
        return category.strip().title() in Validators.VALID_CATEGORIES
    
    @staticmethod
    def validate_investment_type(investment_type: str) -> bool:
        """
        Validate investment type.
        
        Args:
            investment_type: Investment type string to validate
            
        Returns:
            bool: True if valid investment type, False otherwise
        """
        if not isinstance(investment_type, str):
            return False
        return investment_type.strip().title() in Validators.VALID_INVESTMENT_TYPES
    
    @staticmethod
    def validate_stock_symbol(symbol: str) -> bool:
        """
        Validate Indian stock symbol format.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            bool: True if valid stock symbol format, False otherwise
        """
        if not isinstance(symbol, str):
            return False
            
        symbol = symbol.upper().strip()
        # NSE format: SYMBOL.NS or BSE format: SYMBOL.BO
        pattern = r"^[A-Z0-9&-]{1,20}\.(NS|BO)$"
        return re.match(pattern, symbol) is not None
    
    @staticmethod
    def validate_description(description: str, max_length: int = 200) -> bool:
        """
        Validate transaction description.
        
        Args:
            description: Description string to validate
            max_length: Maximum allowed length
            
        Returns:
            bool: True if valid description, False otherwise
        """
        if not isinstance(description, str):
            return False
            
        description = description.strip()
        return 1 <= len(description) <= max_length
    
    @staticmethod
    def validate_password(password: str) -> dict:
        """
        Validate password strength.
        
        Args:
            password: Password string to validate
            
        Returns:
            dict: Validation result with 'valid' boolean and 'errors' list
        """
        if not isinstance(password, str):
            return {'valid': False, 'errors': ['Password must be a string']}
        
        errors = []
        
        if len(password) < 8:
            errors.append('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', password):
            errors.append('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', password):
            errors.append('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', password):
            errors.append('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append('Password must contain at least one special character')
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize user input by removing potentially harmful characters.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            str: Sanitized string
        """
        if not isinstance(input_str, str):
            return ""
        
        # Remove HTML tags and script content
        input_str = re.sub(r'<[^>]*>', '', input_str)
        input_str = re.sub(r'<script.*?</script>', '', input_str, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove potentially harmful characters
        input_str = re.sub(r'[<>"\']', '', input_str)
        
        # Limit length
        return input_str.strip()[:max_length]
    
    @staticmethod
    def validate_json_structure(data: dict, required_fields: List[str]) -> dict:
        """
        Validate JSON structure has required fields.
        
        Args:
            data: Dictionary to validate
            required_fields: List of required field names
            
        Returns:
            dict: Validation result with 'valid' boolean and 'missing_fields' list
        """
        if not isinstance(data, dict):
            return {'valid': False, 'missing_fields': required_fields}
        
        missing_fields = [field for field in required_fields if field not in data]
        return {'valid': len(missing_fields) == 0, 'missing_fields': missing_fields}

# Example usage and testing
if __name__ == "__main__":
    # Test validators
    print("Testing validators...")
    
    # Test date validation
    print(f"Valid date: {Validators.validate_date('2025-06-23')}")  # True
    print(f"Invalid date: {Validators.validate_date('2025-13-45')}")  # False
    
    # Test amount validation
    print(f"Valid amount: {Validators.validate_amount('₹1,50,000')}")  # True
    print(f"Invalid amount: {Validators.validate_amount('-500')}")  # False
    
    # Test city validation
    print(f"Valid city: {Validators.validate_city('Mumbai')}")  # True
    print(f"Invalid city: {Validators.validate_city('InvalidCity')}")  # False
    
    # Test PAN validation
    print(f"Valid PAN: {Validators.validate_pan('ABCDE1234F')}")  # True
    print(f"Invalid PAN: {Validators.validate_pan('INVALID')}")  # False
    
    # Test password validation
    password_result = Validators.validate_password('MyPass123!')
    print(f"Password validation: {password_result}")
    
    print("All tests completed!")
