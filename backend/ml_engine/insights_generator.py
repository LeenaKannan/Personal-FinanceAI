# AI insights engine
# backend/ml_engine/insights_generator.py

from typing import List, Dict

class InsightsGenerator:
    """
    Generates actionable financial insights for the user.
    """

    def __init__(self):
        pass

    def generate(self, user_data: Dict, transactions: List[Dict]) -> List[Dict]:
        """
        Generate insights based on user profile and transactions.
        """
        insights = []
        # Example: High spending on transport
        transport_sum = sum(t["amount"] for t in transactions if t["category"] == "Transport")
        avg_transport = user_data.get("avg_transport", 0)
        if avg_transport and transport_sum > 1.2 * avg_transport:
            insights.append({
                "type": "warning",
                "title": "High Transportation Spending",
                "message": f"You spent {int((transport_sum/avg_transport-1)*100)}% more on transport this month.",
                "action": "Consider using public transport or carpooling."
            })

        # Example: Savings opportunity
        income = user_data.get("income", 0)
        total_spent = sum(t["amount"] for t in transactions if t["amount"] < 0)
        savings_rate = (income + total_spent) / income if income else 0
        if savings_rate > 0.3:
            insights.append({
                "type": "tip",
                "title": "Good Savings Rate",
                "message": f"You saved {int(savings_rate*100)}% of your income this month.",
                "action": "Consider investing more of your savings."
            })

        # Add more rules/ML-based insights here

        return insights
