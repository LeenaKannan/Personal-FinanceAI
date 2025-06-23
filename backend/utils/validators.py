# Input validation
import re
from datetime import datetime

class Validators:
    @staticmethod
    def validate_date(date_str, date_format='%Y-%m-%d'):
        try:
            datetime.strptime(date_str, date_format)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_amount(amount):
        try:
            val = float(amount)
            return val >= 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_city(city, valid_cities):
        return city in valid_cities

    @staticmethod
    def validate_gender(gender):
        return gender.lower() in ['male', 'female', 'other']

    @staticmethod
    def validate_email(email):
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(pattern, email) is not None
