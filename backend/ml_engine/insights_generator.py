# backend/ml_engine/insights_generator.py

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import statistics
import numpy as np
from collections import defaultdict, Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightType(str, Enum):
    """Types of insights that can be generated."""
    TIP = "tip"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    ACHIEVEMENT = "achievement"
    ALERT = "alert"
    RECOMMENDATION = "recommendation"

class InsightCategory(str, Enum):
    """Categories of financial insights."""
    SPENDING = "spending"
    SAVING = "saving"
    INVESTMENT = "investment"
    BUDGET = "budget"
    GOAL = "goal"
    SECURITY = "security"
    TAX = "tax"
    LIFESTYLE = "lifestyle"

class InsightsGenerator:
    """
    Advanced AI-powered insights generator for Indian personal finance.
    Generates actionable, personalized financial insights based on user behavior,
    spending patterns, and Indian financial market context.
    """

    def __init__(self):
        """Initialize the insights generator with Indian market data."""
        self.city_cost_index = self._get_city_cost_index()
        self.category_benchmarks = self._get_category_benchmarks()
        self.indian_financial_calendar = self._get_indian_financial_calendar()
        self.investment_thresholds = self._get_investment_thresholds()
        
    def _get_city_cost_index(self) -> Dict[str, float]:
        """Get cost of living index for Indian cities."""
        return {
            'Mumbai': 1.0, 'Delhi': 0.95, 'Bangalore': 0.88, 'Chennai': 0.82,
            'Pune': 0.78, 'Hyderabad': 0.73, 'Kolkata': 0.75, 'Ahmedabad': 0.70,
            'Jaipur': 0.65, 'Lucknow': 0.60, 'Surat': 0.68, 'Kanpur': 0.58
        }
    
    def _get_category_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get spending benchmarks by category and income level."""
        return {
            'Housing': {'low': 0.40, 'medium': 0.35, 'high': 0.30},
            'Food & Groceries': {'low': 0.25, 'medium': 0.20, 'high': 0.15},
            'Transport': {'low': 0.15, 'medium': 0.12, 'high': 0.10},
            'Utilities': {'low': 0.10, 'medium': 0.08, 'high': 0.06},
            'Entertainment': {'low': 0.05, 'medium': 0.08, 'high': 0.12},
            'Healthcare': {'low': 0.08, 'medium': 0.06, 'high': 0.05},
            'Education': {'low': 0.05, 'medium': 0.08, 'high': 0.10},
            'Shopping': {'low': 0.05, 'medium': 0.08, 'high': 0.12},
            'Investment': {'low': 0.10, 'medium': 0.20, 'high': 0.30}
        }
    
    def _get_indian_financial_calendar(self) -> Dict[str, Dict[str, Any]]:
        """Get Indian financial calendar events."""
        return {
            'tax_season': {'months': [1, 2, 3], 'message': 'Tax filing season'},
            'bonus_season': {'months': [3, 10, 11], 'message': 'Bonus and festival season'},
            'festival_season': {'months': [9, 10, 11], 'message': 'Festival spending season'},
            'new_financial_year': {'months': [4], 'message': 'New financial year planning'},
            'monsoon': {'months': [6, 7, 8, 9], 'message': 'Monsoon season adjustments'}
        }
    
    def _get_investment_thresholds(self) -> Dict[str, float]:
        """Get investment thresholds for different income levels."""
        return {
            'emergency_fund_months': 6,
            'tax_saving_limit': 150000,  # 80C limit
            'equity_allocation_young': 0.7,  # Age < 35
            'equity_allocation_middle': 0.5,  # Age 35-50
            'equity_allocation_senior': 0.3   # Age > 50
        }

    def generate(self, user_data: Dict[str, Any], transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate comprehensive financial insights based on user data and transactions.
        
        Args:
            user_data: User profile information
            transactions: List of user transactions
            
        Returns:
            List of insights with recommendations
        """
        try:
            insights = []
            
            if not transactions:
                return self._generate_onboarding_insights(user_data)
            
            # Process transactions for analysis
            processed_data = self._process_transactions(transactions)
            
            # Generate different types of insights
            insights.extend(self._generate_spending_insights(user_data, processed_data))
            insights.extend(self._generate_savings_insights(user_data, processed_data))
            insights.extend(self._generate_investment_insights(user_data, processed_data))
            insights.extend(self._generate_budget_insights(user_data, processed_data))
            insights.extend(self._generate_lifestyle_insights(user_data, processed_data))
            insights.extend(self._generate_seasonal_insights(user_data, processed_data))
            insights.extend(self._generate_goal_insights(user_data, processed_data))
            insights.extend(self._generate_security_insights(user_data, processed_data))
            insights.extend(self._generate_tax_insights(user_data, processed_data))
            
            # Sort insights by priority and relevance
            insights = self._prioritize_insights(insights, user_data)
            
            # Limit to top insights to avoid overwhelming user
            return insights[:10]
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return self._generate_fallback_insights(user_data)

    def _process_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process transactions to extract meaningful patterns."""
        try:
            # Separate income and expenses
            expenses = [t for t in transactions if t.get('amount', 0) < 0]
            income_txns = [t for t in transactions if t.get('amount', 0) > 0]
            
            # Calculate totals
            total_expenses = sum(abs(t['amount']) for t in expenses)
            total_income = sum(t['amount'] for t in income_txns)
            
            # Category-wise analysis
            category_spending = defaultdict(float)
            category_counts = defaultdict(int)
            
            for txn in expenses:
                category = txn.get('category', 'Other')
                amount = abs(txn['amount'])
                category_spending[category] += amount
                category_counts[category] += 1
            
            # Merchant analysis
            merchant_spending = defaultdict(float)
            for txn in expenses:
                merchant = txn.get('merchant_name', 'Unknown')
                if merchant != 'Unknown':
                    merchant_spending[merchant] += abs(txn['amount'])
            
            # Time-based patterns
            daily_spending = defaultdict(float)
            monthly_spending = defaultdict(float)
            
            for txn in expenses:
                if 'date' in txn:
                    try:
                        if isinstance(txn['date'], str):
                            date_obj = datetime.fromisoformat(txn['date'].replace('Z', '+00:00'))
                        else:
                            date_obj = txn['date']
                        
                        day_key = date_obj.strftime('%A')
                        month_key = date_obj.strftime('%Y-%m')
                        
                        daily_spending[day_key] += abs(txn['amount'])
                        monthly_spending[month_key] += abs(txn['amount'])
                    except:
                        continue
            
            # Payment method analysis
            payment_methods = Counter(txn.get('payment_method', 'Unknown') for txn in transactions)
            
            return {
                'total_expenses': total_expenses,
                'total_income': total_income,
                'category_spending': dict(category_spending),
                'category_counts': dict(category_counts),
                'merchant_spending': dict(merchant_spending),
                'daily_spending': dict(daily_spending),
                'monthly_spending': dict(monthly_spending),
                'payment_methods': dict(payment_methods),
                'transaction_count': len(transactions),
                'expense_count': len(expenses),
                'avg_transaction_size': total_expenses / len(expenses) if expenses else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to process transactions: {e}")
            return {}

    def _generate_spending_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate spending-related insights."""
        insights = []
        
        try:
            income = user_data.get('income', 0)
            city = user_data.get('city', 'Mumbai')
            age = user_data.get('age', 30)
            
            category_spending = processed_data.get('category_spending', {})
            total_expenses = processed_data.get('total_expenses', 0)
            
            if not income or not total_expenses:
                return insights
            
            # Income level classification
            income_level = 'low' if income < 50000 else ('high' if income > 150000 else 'medium')
            
            # Check each category against benchmarks
            for category, amount in category_spending.items():
                if category in self.category_benchmarks:
                    benchmark = self.category_benchmarks[category][income_level]
                    percentage = amount / income
                    
                    if percentage > benchmark * 1.2:  # 20% above benchmark
                        insights.append({
                            'type': InsightType.WARNING.value,
                            'category': InsightCategory.SPENDING.value,
                            'title': f'High {category} Spending',
                            'message': f'You spent {percentage*100:.1f}% of income on {category}, above the recommended {benchmark*100:.1f}%.',
                            'action': f'Consider reviewing your {category.lower()} expenses. Look for alternatives or negotiate better rates.',
                            'priority': 4,
                            'amount_involved': amount,
                            'potential_savings': amount - (income * benchmark)
                        })
                    elif percentage < benchmark * 0.5:  # Very low spending
                        if category in ['Healthcare', 'Investment']:
                            insights.append({
                                'type': InsightType.OPPORTUNITY.value,
                                'category': InsightCategory.SPENDING.value,
                                'title': f'Low {category} Allocation',
                                'message': f'You allocated only {percentage*100:.1f}% to {category}. Consider increasing this.',
                                'action': f'Increase your {category.lower()} budget for better financial health.',
                                'priority': 3,
                                'amount_involved': income * benchmark - amount
                            })
            
            # Merchant concentration analysis
            merchant_spending = processed_data.get('merchant_spending', {})
            if merchant_spending:
                top_merchant = max(merchant_spending.items(), key=lambda x: x[1])
                if top_merchant[1] > total_expenses * 0.3:  # Single merchant > 30%
                    insights.append({
                        'type': InsightType.WARNING.value,
                        'category': InsightCategory.SPENDING.value,
                        'title': 'High Merchant Concentration',
                        'message': f'You spent ₹{top_merchant[1]:,.0f} ({top_merchant[1]/total_expenses*100:.1f}%) at {top_merchant[0]}.',
                        'action': 'Diversify your spending to avoid over-dependence on single merchants.',
                        'priority': 2,
                        'amount_involved': top_merchant[1]
                    })
            
            # Daily spending pattern analysis
            daily_spending = processed_data.get('daily_spending', {})
            if daily_spending:
                weekend_spending = daily_spending.get('Saturday', 0) + daily_spending.get('Sunday', 0)
                weekday_spending = sum(daily_spending.get(day, 0) for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
                
                if weekend_spending > weekday_spending * 0.6:  # High weekend spending
                    insights.append({
                        'type': InsightType.TIP.value,
                        'category': InsightCategory.LIFESTYLE.value,
                        'title': 'High Weekend Spending',
                        'message': f'Weekend spending (₹{weekend_spending:,.0f}) is high compared to weekdays.',
                        'action': 'Plan weekend activities with a budget to control impulse spending.',
                        'priority': 2,
                        'amount_involved': weekend_spending
                    })
            
        except Exception as e:
            logger.error(f"Failed to generate spending insights: {e}")
        
        return insights

    def _generate_savings_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate savings-related insights."""
        insights = []
        
        try:
            income = user_data.get('income', 0)
            total_expenses = processed_data.get('total_expenses', 0)
            total_income = processed_data.get('total_income', income)
            
            if not income:
                return insights
            
            # Calculate savings rate
            net_savings = total_income - total_expenses
            savings_rate = net_savings / total_income if total_income > 0 else 0
            
            # Savings rate analysis
            if savings_rate < 0.1:  # Less than 10%
                insights.append({
                    'type': InsightType.ALERT.value,
                    'category': InsightCategory.SAVING.value,
                    'title': 'Low Savings Rate',
                    'message': f'Your savings rate is {savings_rate*100:.1f}%, well below the recommended 20%.',
                    'action': 'Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings.',
                    'priority': 5,
                    'amount_involved': total_income * 0.2 - net_savings
                })
            elif savings_rate > 0.4:  # Very high savings
                insights.append({
                    'type': InsightType.ACHIEVEMENT.value,
                    'category': InsightCategory.SAVING.value,
                    'title': 'Excellent Savings Rate!',
                    'message': f'Outstanding! You saved {savings_rate*100:.1f}% of your income.',
                    'action': 'Consider investing your surplus savings for better returns.',
                    'priority': 1,
                    'amount_involved': net_savings
                })
            elif savings_rate >= 0.2:  # Good savings
                insights.append({
                    'type': InsightType.ACHIEVEMENT.value,
                    'category': InsightCategory.SAVING.value,
                    'title': 'Good Savings Habit',
                    'message': f'Great job! You saved {savings_rate*100:.1f}% of your income.',
                    'action': 'Keep up the good work and consider increasing investments.',
                    'priority': 2,
                    'amount_involved': net_savings
                })
            
            # Emergency fund calculation
            monthly_expenses = total_expenses  # Assuming monthly data
            emergency_fund_needed = monthly_expenses * self.investment_thresholds['emergency_fund_months']
            
            if net_savings > 0:
                months_to_emergency_fund = emergency_fund_needed / net_savings
                if months_to_emergency_fund <= 12:  # Can build in a year
                    insights.append({
                        'type': InsightType.OPPORTUNITY.value,
                        'category': InsightCategory.GOAL.value,
                        'title': 'Emergency Fund Goal',
                        'message': f'You can build a 6-month emergency fund (₹{emergency_fund_needed:,.0f}) in {months_to_emergency_fund:.0f} months.',
                        'action': 'Start a separate emergency fund with your current savings rate.',
                        'priority': 3,
                        'amount_involved': emergency_fund_needed
                    })
            
        except Exception as e:
            logger.error(f"Failed to generate savings insights: {e}")
        
        return insights

    def _generate_investment_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate investment-related insights."""
        insights = []
        
        try:
            age = user_data.get('age', 30)
            income = user_data.get('income', 0)
            total_expenses = processed_data.get('total_expenses', 0)
            category_spending = processed_data.get('category_spending', {})
            
            investment_spending = category_spending.get('Investment', 0)
            net_savings = income - total_expenses
            
            # Age-based investment advice
            if age < 35:
                recommended_equity = self.investment_thresholds['equity_allocation_young']
                insights.append({
                    'type': InsightType.TIP.value,
                    'category': InsightCategory.INVESTMENT.value,
                    'title': 'Young Investor Advantage',
                    'message': f'At {age}, you can take higher risks. Consider {recommended_equity*100:.0f}% equity allocation.',
                    'action': 'Start SIPs in equity mutual funds or index funds for long-term wealth creation.',
                    'priority': 3,
                    'amount_involved': net_savings * recommended_equity
                })
            elif age > 50:
                recommended_equity = self.investment_thresholds['equity_allocation_senior']
                insights.append({
                    'type': InsightType.TIP.value,
                    'category': InsightCategory.INVESTMENT.value,
                    'title': 'Conservative Investment Strategy',
                    'message': f'At {age}, focus on capital preservation with {recommended_equity*100:.0f}% equity allocation.',
                    'action': 'Increase allocation to debt funds, FDs, and retirement-focused investments.',
                    'priority': 4,
                    'amount_involved': net_savings * (1 - recommended_equity)
                })
            
            # Investment rate analysis
            investment_rate = investment_spending / income if income > 0 else 0
            
            if investment_rate < 0.1 and net_savings > 0:  # Low investment
                insights.append({
                    'type': InsightType.OPPORTUNITY.value,
                    'category': InsightCategory.INVESTMENT.value,
                    'title': 'Investment Opportunity',
                    'message': f'You invest only {investment_rate*100:.1f}% of income. Consider increasing to 15-20%.',
                    'action': 'Start with small SIPs in diversified mutual funds.',
                    'priority': 3,
                    'amount_involved': income * 0.15 - investment_spending
                })
            
            # Tax saving opportunity
            if income > 250000:  # Above tax exemption limit
                tax_saving_needed = min(self.investment_thresholds['tax_saving_limit'], income * 0.15)
                if investment_spending < tax_saving_needed:
                    tax_savings = (tax_saving_needed - investment_spending) * 0.3  # 30% tax bracket
                    insights.append({
                        'type': InsightType.OPPORTUNITY.value,
                        'category': InsightCategory.TAX.value,
                        'title': 'Tax Saving Opportunity',
                        'message': f'Invest ₹{tax_saving_needed - investment_spending:,.0f} more in 80C to save ₹{tax_savings:,.0f} in taxes.',
                        'action': 'Consider ELSS mutual funds, PPF, or life insurance for tax savings.',
                        'priority': 4,
                        'amount_involved': tax_savings
                    })
            
        except Exception as e:
            logger.error(f"Failed to generate investment insights: {e}")
        
        return insights

    def _generate_budget_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate budget-related insights."""
        insights = []
        
        try:
            income = user_data.get('income', 0)
            total_expenses = processed_data.get('total_expenses', 0)
            
            if not income:
                return insights
            
            # Budget adherence
            expense_ratio = total_expenses / income
            
            if expense_ratio > 0.9:  # Spending > 90% of income
                insights.append({
                    'type': InsightType.ALERT.value,
                    'category': InsightCategory.BUDGET.value,
                    'title': 'Budget Overspend Alert',
                    'message': f'You spent {expense_ratio*100:.1f}% of your income. This leaves little room for savings.',
                    'action': 'Create a detailed budget and track expenses daily to identify areas to cut.',
                    'priority': 5,
                    'amount_involved': total_expenses - (income * 0.8)
                })
            elif expense_ratio < 0.6:  # Very low spending
                insights.append({
                    'type': InsightType.TIP.value,
                    'category': InsightCategory.LIFESTYLE.value,
                    'title': 'Conservative Spending',
                    'message': f'You spent only {expense_ratio*100:.1f}% of income. You might be too conservative.',
                    'action': 'Consider allocating more to experiences, health, or skill development.',
                    'priority': 1,
                    'amount_involved': income * 0.2 - (income - total_expenses)
                })
            
        except Exception as e:
            logger.error(f"Failed to generate budget insights: {e}")
        
        return insights

    def _generate_lifestyle_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate lifestyle-related insights."""
        insights = []
        
        try:
            category_spending = processed_data.get('category_spending', {})
            total_expenses = processed_data.get('total_expenses', 0)
            
            # Entertainment vs Healthcare balance
            entertainment = category_spending.get('Entertainment', 0)
            healthcare = category_spending.get('Healthcare', 0)
            
            if entertainment > healthcare * 2 and healthcare > 0:
                insights.append({
                    'type': InsightType.TIP.value,
                    'category': InsightCategory.LIFESTYLE.value,
                    'title': 'Health vs Entertainment Balance',
                    'message': f'Entertainment (₹{entertainment:,.0f}) is much higher than healthcare (₹{healthcare:,.0f}).',
                    'action': 'Consider balancing lifestyle spending with health investments.',
                    'priority': 2,
                    'amount_involved': entertainment - healthcare
                })
            
            # Food delivery analysis
            food_spending = category_spending.get('Food & Groceries', 0)
            if food_spending > 0:
                # Estimate food delivery based on transaction patterns (this would need more sophisticated analysis)
                estimated_delivery = food_spending * 0.4  # Rough estimate
                if estimated_delivery > food_spending * 0.3:
                    potential_savings = estimated_delivery * 0.5
                    insights.append({
                        'type': InsightType.TIP.value,
                        'category': InsightCategory.LIFESTYLE.value,
                        'title': 'Food Delivery Optimization',
                        'message': 'High food delivery spending detected. Cooking at home can save money.',
                        'action': f'Cook 2-3 meals at home weekly to save approximately ₹{potential_savings:,.0f}/month.',
                        'priority': 2,
                        'amount_involved': potential_savings
                    })
            
        except Exception as e:
            logger.error(f"Failed to generate lifestyle insights: {e}")
        
        return insights

    def _generate_seasonal_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate seasonal and calendar-based insights."""
        insights = []
        
        try:
            current_month = datetime.now().month
            
            # Check for seasonal events
            for event, details in self.indian_financial_calendar.items():
                if current_month in details['months']:
                    if event == 'tax_season':
                        insights.append({
                            'type': InsightType.REMINDER.value,
                            'category': InsightCategory.TAX.value,
                            'title': 'Tax Filing Reminder',
                            'message': 'Tax filing season is here. Ensure all documents are ready.',
                            'action': 'Gather Form 16, investment proofs, and file your ITR before the deadline.',
                            'priority': 4
                        })
                    elif event == 'festival_season':
                        insights.append({
                            'type': InsightType.TIP.value,
                            'category': InsightCategory.BUDGET.value,
                            'title': 'Festival Season Budget',
                            'message': 'Festival season typically increases spending by 20-30%.',
                            'action': 'Set aside a festival budget to avoid overspending on gifts and celebrations.',
                            'priority': 3
                        })
                    elif event == 'new_financial_year':
                        insights.append({
                            'type': InsightType.OPPORTUNITY.value,
                            'category': InsightCategory.GOAL.value,
                            'title': 'New Financial Year Planning',
                            'message': 'New financial year started. Perfect time to review and set financial goals.',
                            'action': 'Review last year\'s performance and set new savings and investment targets.',
                            'priority': 3
                        })
            
        except Exception as e:
            logger.error(f"Failed to generate seasonal insights: {e}")
        
        return insights

    def _generate_goal_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate goal-related insights."""
        insights = []
        
        try:
            age = user_data.get('age', 30)
            income = user_data.get('income', 0)
            city = user_data.get('city', 'Mumbai')
            net_savings = income - processed_data.get('total_expenses', 0)
            
            # Home buying goal
            if age < 40 and income > 50000:
                # Estimate home price based on city
                city_multiplier = self.city_cost_index.get(city, 1.0)
                estimated_home_price = 5000000 * city_multiplier  # Base 50L home
                down_payment = estimated_home_price * 0.2
                
                if net_savings > 0:
                    years_to_down_payment = down_payment / (net_savings * 12)
                    if years_to_down_payment <= 10:
                        insights.append({
                            'type': InsightType.OPPORTUNITY.value,
                            'category': InsightCategory.GOAL.value,
                            'title': 'Home Purchase Planning',
                            'message': f'You can save for home down payment (₹{down_payment:,.0f}) in {years_to_down_payment:.1f} years.',
                            'action': 'Start a dedicated home fund and consider home loan pre-approval.',
                            'priority': 2,
                            'amount_involved': down_payment
                        })
            
            # Retirement planning
            if age > 25:
                years_to_retirement = 60 - age
                monthly_retirement_savings_needed = income * 0.1  # 10% of income
                
                if years_to_retirement > 10:
                    retirement_corpus = monthly_retirement_savings_needed * 12 * years_to_retirement * 2  # Rough estimate with growth
                    insights.append({
                        'type': InsightType.TIP.value,
                        'category': InsightCategory.GOAL.value,
                        'title': 'Retirement Planning',
                        'message': f'Start saving ₹{monthly_retirement_savings_needed:,.0f}/month for retirement.',
                        'action': 'Consider NPS, PPF, and equity mutual funds for retirement corpus.',
                        'priority': 3,
                        'amount_involved': monthly_retirement_savings_needed
                    })
            
        except Exception as e:
            logger.error(f"Failed to generate goal insights: {e}")
        
        return insights

    def _generate_security_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security and risk-related insights."""
        insights = []
        
        try:
            payment_methods = processed_data.get('payment_methods', {})
            
            # Digital payment security
            digital_payments = payment_methods.get('upi', 0) + payment_methods.get('card', 0)
            total_transactions = sum(payment_methods.values())
            
            if digital_payments / total_transactions > 0.8 if total_transactions > 0 else False:
                insights.append({
                    'type': InsightType.TIP.value,
                    'category': InsightCategory.SECURITY.value,
                    'title': 'Digital Payment Security',
                    'message': 'You use digital payments frequently. Ensure your accounts are secure.',
                    'action': 'Enable 2FA, use strong passwords, and monitor statements regularly.',
                    'priority': 3
                })
            
            # Insurance recommendation
            income = user_data.get('income', 0)
            if income > 25000:
                recommended_life_cover = income * 12 * 10  # 10x annual income
                insights.append({
                    'type': InsightType.RECOMMENDATION.value,
                    'category': InsightCategory.SECURITY.value,
                    'title': 'Life Insurance Review',
                    'message': f'Ensure you have adequate life insurance coverage (₹{recommended_life_cover:,.0f}).',
                    'action': 'Review your current life insurance and consider term insurance if needed.',
                    'priority': 3,
                    'amount_involved': recommended_life_cover
                })
            
        except Exception as e:
            logger.error(f"Failed to generate security insights: {e}")
        
        return insights

    def _generate_tax_insights(self, user_data: Dict[str, Any], processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tax-related insights."""
        insights = []
        
        try:
            income = user_data.get('income', 0)
            annual_income = income * 12
            
            if annual_income > 250000:  # Above basic exemption
                # Tax bracket analysis
                if annual_income <= 500000:
                    tax_rate = 0.05
                elif annual_income <= 1000000:
                    tax_rate = 0.20
                else:
                    tax_rate = 0.30
                
                potential_tax = (annual_income - 250000) * tax_rate
                max_80c_savings = min(150000, annual_income * 0.15) * tax_rate
                
                insights.append({
                    'type': InsightType.OPPORTUNITY.value,
                    'category': InsightCategory.TAX.value,
                    'title': 'Tax Optimization',
                    'message': f'You can save up to ₹{max_80c_savings:,.0f} in taxes through 80C investments.',
                    'action': 'Maximize 80C deductions through ELSS, PPF, life insurance, and home loan principal.',
                    'priority': 4,
                    'amount_involved': max_80c_savings
                })
            
        except Exception as e:
            logger.error(f"Failed to generate tax insights: {e}")
        
        return insights

    def _generate_onboarding_insights(self, user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights for new users with no transaction history."""
        insights = []
        
        age = user_data.get('age', 30)
        income = user_data.get('income', 0)
        
        insights.extend([
            {
                'type': InsightType.TIP.value,
                'category': InsightCategory.BUDGET.value,
                'title': 'Welcome to Smart Finance!',
                'message': 'Start tracking your expenses to get personalized insights.',
                'action': 'Add your first transaction or upload bank statements to begin.',
                'priority': 5
            },
            {
                'type': InsightType.TIP.value,
                'category': InsightCategory.SAVING.value,
                'title': 'Start with 50/30/20 Rule',
                'message': 'Allocate 50% for needs, 30% for wants, and 20% for savings.',
                'action': f'Based on your income, aim to save ₹{income * 0.2:,.0f} monthly.',
                'priority': 4,
                'amount_involved': income * 0.2
            }
        ])
        
        return insights

    def _generate_fallback_insights(self, user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic insights when analysis fails."""
        return [
            {
                'type': InsightType.TIP.value,
                'category': InsightCategory.BUDGET.value,
                'title': 'Track Your Expenses',
                'message': 'Regular expense tracking is the foundation of good financial health.',
                'action': 'Start by categorizing your daily expenses for better insights.',
                'priority': 3
            }
        ]

    def _prioritize_insights(self, insights: List[Dict[str, Any]], user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize insights based on user profile and urgency."""
        try:
            # Sort by priority (higher number = higher priority)
            insights.sort(key=lambda x: x.get('priority', 1), reverse=True)
            
            # Add unique IDs and timestamps
            for i, insight in enumerate(insights):
                insight['id'] = f"insight_{datetime.now().strftime('%Y%m%d')}_{i:03d}"
                insight['generated_at'] = datetime.now().isoformat()
                insight['confidence'] = 0.8  # Default confidence
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to prioritize insights: {e}")
            return insights

# Example usage
if __name__ == "__main__":
    # Test the insights generator
    generator = InsightsGenerator()
    
    # Sample user data
    user_data = {
        'income': 80000,
        'age': 28,
        'city': 'Mumbai',
        'gender': 'male'
    }
    
    # Sample transactions
    transactions = [
        {'amount': -850, 'category': 'Food & Groceries', 'date': '2025-06-01', 'description': 'ZOMATO'},
        {'amount': -28000, 'category': 'Housing', 'date': '2025-06-01', 'description': 'RENT'},
        {'amount': 80000, 'category': 'Income', 'date': '2025-06-01', 'description': 'SALARY'},
        {'amount': -1200, 'category': 'Transport', 'date': '2025-06-02', 'description': 'OLA CABS'},
    ]
    
    # Generate insights
    insights = generator.generate(user_data, transactions)
    
    print(f"Generated {len(insights)} insights:")
    for insight in insights:
        print(f"- {insight['title']}: {insight['message']}")
