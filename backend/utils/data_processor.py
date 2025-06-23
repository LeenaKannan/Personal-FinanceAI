# Data processing utilities
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, inflation_rate=0.06, city_cost_index=None):
        self.inflation_rate = inflation_rate
        self.city_cost_index = city_cost_index or {}

    def adjust_for_inflation(self, amount, years=1):
        return amount * ((1 + self.inflation_rate) ** years)

    def adjust_for_city_cost(self, amount, city):
        base_index = self.city_cost_index.get('Mumbai', 100)
        city_index = self.city_cost_index.get(city, base_index)
        return amount * (city_index / base_index)

    def preprocess_transactions(self, transactions_df):
        df = transactions_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.fillna(0, inplace=True)
        return df

    def categorize_expenses(self, transactions_df, categorizer):
        df = transactions_df.copy()
        df['category'] = df['description'].apply(categorizer)
        return df

    def aggregate_expenses(self, transactions_df):
        df = transactions_df.copy()
        return df.groupby('category')['amount'].sum().to_dict()

    def predict_expenses(self, user_profile, model):
        features = [user_profile.get('income', 0),
                    self.city_cost_index.get(user_profile.get('city', 'Mumbai'), 100),
                    1 if user_profile.get('gender', 'male') == 'male' else 0,
                    user_profile.get('age', 30)]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return prediction
