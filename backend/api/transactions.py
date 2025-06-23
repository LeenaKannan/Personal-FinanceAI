# backend/api/transactions.py

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import logging
import csv
import io

from backend.models import transaction, user
from backend.models.database import get_db
from backend.api.auth import get_current_user
from backend.ml_engine.categorizer import TransactionCategorizer
from backend.utils.pdf_parser import PDFParser
from backend.utils.data_processor import DataProcessor
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/transactions", tags=["transactions"])

# Initialize components
categorizer = TransactionCategorizer()
pdf_parser = PDFParser()
data_processor = DataProcessor()

# === ENUMS ===

class SortOrderEnum(str, Enum):
    ASC = "asc"
    DESC = "desc"

class ExportFormatEnum(str, Enum):
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"

# === ADDITIONAL SCHEMAS ===

class TransactionBulkCreate(BaseModel):
    """Schema for bulk transaction creation."""
    
    transactions: List[transaction.TransactionCreate] = Field(
        ..., 
        min_items=1, 
        max_items=1000, 
        description="List of transactions to create"
    )

class TransactionStats(BaseModel):
    """Schema for transaction statistics."""
    
    total_transactions: int
    total_income: float
    total_expenses: float
    net_amount: float
    average_transaction_size: float
    largest_transaction: float
    categories_count: int
    date_range: Dict[str, str]

class TransactionAnalytics(BaseModel):
    """Schema for transaction analytics."""
    
    monthly_trends: Dict[str, float]
    category_breakdown: Dict[str, float]
    payment_method_distribution: Dict[str, int]
    spending_patterns: Dict[str, Any]
    top_merchants: List[Dict[str, Any]]

class BankStatementUpload(BaseModel):
    """Schema for bank statement upload response."""
    
    total_transactions: int
    successful_imports: int
    failed_imports: int
    duplicate_transactions: int
    imported_transactions: List[transaction.TransactionOut]
    errors: List[str]

# === UTILITY FUNCTIONS ===

def apply_transaction_filters(
    query, 
    filters: Optional[transaction.TransactionFilter] = None
):
    """Apply filters to transaction query."""
    if not filters:
        return query
    
    if filters.start_date:
        query = query.filter(transaction.Transaction.date >= filters.start_date)
    if filters.end_date:
        query = query.filter(transaction.Transaction.date <= filters.end_date)
    if filters.category:
        query = query.filter(transaction.Transaction.category == filters.category)
    if filters.payment_method:
        query = query.filter(transaction.Transaction.payment_method == filters.payment_method)
    if filters.min_amount:
        query = query.filter(transaction.Transaction.amount >= filters.min_amount)
    if filters.max_amount:
        query = query.filter(transaction.Transaction.amount <= filters.max_amount)
    if filters.transaction_type:
        query = query.filter(transaction.Transaction.transaction_type == filters.transaction_type)
    if filters.city:
        query = query.filter(transaction.Transaction.city.ilike(f"%{filters.city}%"))
    if filters.is_business is not None:
        query = query.filter(transaction.Transaction.is_business == filters.is_business)
    if filters.search_term:
        search = f"%{filters.search_term}%"
        query = query.filter(
            transaction.Transaction.description.ilike(search) |
            transaction.Transaction.merchant_name.ilike(search) |
            transaction.Transaction.notes.ilike(search)
        )
    
    return query

def validate_transaction_ownership(txn: transaction.Transaction, user_id: int):
    """Validate that transaction belongs to the current user."""
    if not txn or txn.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found"
        )

# === API ENDPOINTS ===

@router.post("/", response_model=transaction.TransactionOut, status_code=status.HTTP_201_CREATED)
async def add_transaction(
    txn_in: transaction.TransactionCreate,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    """
    Create a new transaction with automatic categorization.
    """
    try:
        # Auto-categorize if not provided or category is 'Other'
        if not txn_in.category or txn_in.category == transaction.CategoryEnum.OTHER:
            predicted_category = categorizer.categorize(txn_in.description)
            txn_in.category = transaction.CategoryEnum(predicted_category)
        
        # Set user ID
        txn_in.user_id = current_user.id
        
        # Create transaction
        new_transaction = transaction.create_transaction(db, txn_in)
        
        logger.info(f"Created transaction {new_transaction.id} for user {current_user.id}")
        
        return new_transaction
        
    except Exception as e:
        logger.error(f"Failed to create transaction for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create transaction"
        )

@router.post("/bulk", response_model=List[transaction.TransactionOut], status_code=status.HTTP_201_CREATED)
async def add_bulk_transactions(
    bulk_data: TransactionBulkCreate,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    """
    Create multiple transactions in bulk.
    """
    try:
        created_transactions = []
        
        for txn_data in bulk_data.transactions:
            # Auto-categorize if needed
            if not txn_data.category or txn_data.category == transaction.CategoryEnum.OTHER:
                predicted_category = categorizer.categorize(txn_data.description)
                txn_data.category = transaction.CategoryEnum(predicted_category)
            
            # Set user ID
            txn_data.user_id = current_user.id
        
        # Bulk create
        created_transactions = transaction.bulk_create_transactions(db, bulk_data.transactions)
        
        logger.info(f"Created {len(created_transactions)} transactions for user {current_user.id}")
        
        return created_transactions
        
    except Exception as e:
        logger.error(f"Failed to create bulk transactions for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create bulk transactions"
        )

@router.get("/", response_model=List[transaction.TransactionOut])
async def list_transactions(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date"),
    category: Optional[transaction.CategoryEnum] = Query(None, description="Filter by category"),
    payment_method: Optional[transaction.PaymentMethodEnum] = Query(None, description="Filter by payment method"),
    min_amount: Optional[float] = Query(None, description="Minimum amount filter"),
    max_amount: Optional[float] = Query(None, description="Maximum amount filter"),
    transaction_type: Optional[transaction.TransactionTypeEnum] = Query(None, description="Filter by transaction type"),
    search: Optional[str] = Query(None, description="Search in description, merchant, notes"),
    sort_by: str = Query("date", description="Sort field"),
    sort_order: SortOrderEnum = Query(SortOrderEnum.DESC, description="Sort order")
):
    """
    Get paginated list of transactions with filtering and sorting.
    """
    try:
        # Create filter object
        filters = transaction.TransactionFilter(
            start_date=start_date,
            end_date=end_date,
            category=category,
            payment_method=payment_method,
            min_amount=min_amount,
            max_amount=max_amount,
            transaction_type=transaction_type,
            search_term=search
        )
        
        # Get transactions
        transactions = transaction.get_transactions_by_user(
            db, current_user.id, skip=skip, limit=limit, filters=filters
        )
        
        # Apply sorting
        if sort_by == "amount":
            transactions.sort(key=lambda x: x.amount, reverse=(sort_order == SortOrderEnum.DESC))
        elif sort_by == "date":
            transactions.sort(key=lambda x: x.date, reverse=(sort_order == SortOrderEnum.DESC))
        
        return transactions
        
    except Exception as e:
        logger.error(f"Failed to list transactions for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transactions"
        )

@router.get("/summary", response_model=transaction.TransactionSummary)
async def get_transaction_summary(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date")
):
    """
    Get transaction summary statistics.
    """
    try:
        filters = transaction.TransactionFilter(
            start_date=start_date,
            end_date=end_date
        )
        
        summary = transaction.get_transaction_summary(db, current_user.id, filters)
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get transaction summary for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate transaction summary"
        )

@router.get("/analytics", response_model=TransactionAnalytics)
async def get_transaction_analytics(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
    months: int = Query(12, ge=1, le=36, description="Number of months to analyze")
):
    """
    Get detailed transaction analytics and insights.
    """
    try:
        # Get transactions for the specified period
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months * 30)
        
        filters = transaction.TransactionFilter(
            start_date=start_date,
            end_date=end_date
        )
        
        transactions = transaction.get_transactions_by_user(
            db, current_user.id, limit=10000, filters=filters
        )
        
        # Calculate analytics
        monthly_trends = {}
        category_breakdown = {}
        payment_method_distribution = {}
        merchant_spending = {}
        
        for txn in transactions:
            # Monthly trends
            month_key = txn.date.strftime('%Y-%m')
            if txn.amount < 0:  # Expenses only
                monthly_trends[month_key] = monthly_trends.get(month_key, 0) + abs(txn.amount)
            
            # Category breakdown
            if txn.amount < 0:  # Expenses only
                category = txn.category.value if hasattr(txn.category, 'value') else str(txn.category)
                category_breakdown[category] = category_breakdown.get(category, 0) + abs(txn.amount)
            
            # Payment method distribution
            if txn.payment_method:
                method = txn.payment_method.value if hasattr(txn.payment_method, 'value') else str(txn.payment_method)
                payment_method_distribution[method] = payment_method_distribution.get(method, 0) + 1
            
            # Merchant spending
            if txn.merchant_name and txn.amount < 0:
                merchant_spending[txn.merchant_name] = merchant_spending.get(txn.merchant_name, 0) + abs(txn.amount)
        
        # Top merchants
        top_merchants = [
            {"merchant": merchant, "total_spent": amount}
            for merchant, amount in sorted(merchant_spending.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Spending patterns
        spending_patterns = {
            "avg_daily_spending": sum(category_breakdown.values()) / (months * 30),
            "highest_category": max(category_breakdown.items(), key=lambda x: x[1])[0] if category_breakdown else None,
            "most_used_payment_method": max(payment_method_distribution.items(), key=lambda x: x[1])[0] if payment_method_distribution else None
        }
        
        return TransactionAnalytics(
            monthly_trends=monthly_trends,
            category_breakdown=category_breakdown,
            payment_method_distribution=payment_method_distribution,
            spending_patterns=spending_patterns,
            top_merchants=top_merchants
        )
        
    except Exception as e:
        logger.error(f"Failed to get transaction analytics for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate transaction analytics"
        )

@router.get("/{txn_id}", response_model=transaction.TransactionOut)
async def get_transaction(
    txn_id: int,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    """
    Get a specific transaction by ID.
    """
    try:
        txn = transaction.get_transaction(db, txn_id)
        validate_transaction_ownership(txn, current_user.id)
        
        return txn
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transaction {txn_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction"
        )

@router.put("/{txn_id}", response_model=transaction.TransactionOut)
async def update_transaction(
    txn_id: int,
    txn_update: transaction.TransactionUpdate,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    """
    Update an existing transaction.
    """
    try:
        # Get existing transaction
        existing_txn = transaction.get_transaction(db, txn_id)
        validate_transaction_ownership(existing_txn, current_user.id)
        
        # Re-categorize if description changed and category is not provided
        if txn_update.description and not txn_update.category:
            predicted_category = categorizer.categorize(txn_update.description)
            txn_update.category = transaction.CategoryEnum(predicted_category)
        
        # Update transaction
        updated_txn = transaction.update_transaction(db, txn_id, txn_update)
        
        logger.info(f"Updated transaction {txn_id} for user {current_user.id}")
        
        return updated_txn
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update transaction {txn_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update transaction"
        )

@router.delete("/{txn_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transaction(
    txn_id: int,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    """
    Delete a transaction.
    """
    try:
        # Get existing transaction
        existing_txn = transaction.get_transaction(db, txn_id)
        validate_transaction_ownership(existing_txn, current_user.id)
        
        # Delete transaction
        success = transaction.delete_transaction(db, txn_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found"
            )
        
        logger.info(f"Deleted transaction {txn_id} for user {current_user.id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete transaction {txn_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete transaction"
        )

@router.post("/upload/bank-statement", response_model=BankStatementUpload)
async def upload_bank_statement(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    """
    Upload and parse bank statement (PDF or CSV).
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            # Handle PDF bank statement
            content = await file.read()
            
            # Save temporarily and parse
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract text and parse transactions
                text = pdf_parser.extract_text(tmp_file_path)
                transactions_df = pdf_parser.parse_transactions(text)
                
                # Convert to transaction objects
                imported_transactions = []
                errors = []
                
                for _, row in transactions_df.iterrows():
                    try:
                        # Auto-categorize
                        category = categorizer.categorize(row['description'])
                        
                        txn_data = transaction.TransactionCreate(
                            user_id=current_user.id,
                            date=row['date'],
                            description=row['description'],
                            category=transaction.CategoryEnum(category),
                            amount=row['amount']
                        )
                        
                        new_txn = transaction.create_transaction(db, txn_data)
                        imported_transactions.append(new_txn)
                        
                    except Exception as e:
                        errors.append(f"Failed to import transaction: {str(e)}")
                
                return BankStatementUpload(
                    total_transactions=len(transactions_df),
                    successful_imports=len(imported_transactions),
                    failed_imports=len(errors),
                    duplicate_transactions=0,  # TODO: Implement duplicate detection
                    imported_transactions=imported_transactions,
                    errors=errors
                )
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(tmp_file_path)
        
        elif file_extension == 'csv':
            # Handle CSV file
            content = await file.read()
            csv_data = content.decode('utf-8')
            
            imported_transactions = []
            errors = []
            
            csv_reader = csv.DictReader(io.StringIO(csv_data))
            
            for row in csv_reader:
                try:
                    # Map CSV columns (adjust based on your CSV format)
                    description = row.get('description', row.get('Description', ''))
                    amount = float(row.get('amount', row.get('Amount', 0)))
                    date_str = row.get('date', row.get('Date', ''))
                    
                    # Parse date
                    txn_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Auto-categorize
                    category = categorizer.categorize(description)
                    
                    txn_data = transaction.TransactionCreate(
                        user_id=current_user.id,
                        date=txn_date,
                        description=description,
                        category=transaction.CategoryEnum(category),
                        amount=amount
                    )
                    
                    new_txn = transaction.create_transaction(db, txn_data)
                    imported_transactions.append(new_txn)
                    
                except Exception as e:
                    errors.append(f"Failed to import row: {str(e)}")
            
            return BankStatementUpload(
                total_transactions=len(list(csv.DictReader(io.StringIO(csv_data)))),
                successful_imports=len(imported_transactions),
                failed_imports=len(errors),
                duplicate_transactions=0,
                imported_transactions=imported_transactions,
                errors=errors
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Please upload PDF or CSV files."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload bank statement for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process bank statement"
        )

@router.get("/export/{format}", response_class=None)
async def export_transactions(
    format: ExportFormatEnum,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
    start_date: Optional[date] = Query(None, description="Filter by start date"),
    end_date: Optional[date] = Query(None, description="Filter by end date")
):
    """
    Export transactions in various formats.
    """
    try:
        # Get transactions
        filters = transaction.TransactionFilter(
            start_date=start_date,
            end_date=end_date
        )
        
        transactions = transaction.get_transactions_by_user(
            db, current_user.id, limit=10000, filters=filters
        )
        
        if format == ExportFormatEnum.CSV:
            # Generate CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Date', 'Description', 'Category', 'Amount', 'Payment Method', 'Merchant'])
            
            # Write data
            for txn in transactions:
                writer.writerow([
                    txn.date,
                    txn.description,
                    txn.category.value if hasattr(txn.category, 'value') else str(txn.category),
                    txn.amount,
                    txn.payment_method.value if txn.payment_method and hasattr(txn.payment_method, 'value') else '',
                    txn.merchant_name or ''
                ])
            
            from fastapi.responses import StreamingResponse
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=transactions_{datetime.now().strftime('%Y%m%d')}.csv"}
            )
        
        elif format == ExportFormatEnum.JSON:
            # Generate JSON
            import json
            from fastapi.responses import JSONResponse
            
            transactions_data = [
                {
                    "id": txn.id,
                    "date": txn.date.isoformat(),
                    "description": txn.description,
                    "category": txn.category.value if hasattr(txn.category, 'value') else str(txn.category),
                    "amount": txn.amount,
                    "payment_method": txn.payment_method.value if txn.payment_method and hasattr(txn.payment_method, 'value') else None,
                    "merchant_name": txn.merchant_name
                }
                for txn in transactions
            ]
            
            return JSONResponse(
                content=transactions_data,
                headers={"Content-Disposition": f"attachment; filename=transactions_{datetime.now().strftime('%Y%m%d')}.json"}
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported export format"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export transactions for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export transactions"
        )

@router.post("/{txn_id}/recategorize", response_model=transaction.TransactionOut)
async def recategorize_transaction(
    txn_id: int,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    """
    Recategorize a transaction using the latest ML model.
    """
    try:
        # Get existing transaction
        existing_txn = transaction.get_transaction(db, txn_id)
        validate_transaction_ownership(existing_txn, current_user.id)
        
        # Recategorize
        new_category = categorizer.categorize(existing_txn.description)
        
        # Update transaction
        txn_update = transaction.TransactionUpdate(
            category=transaction.CategoryEnum(new_category)
        )
        
        updated_txn = transaction.update_transaction(db, txn_id, txn_update)
        
        logger.info(f"Recategorized transaction {txn_id} for user {current_user.id}")
        
        return updated_txn
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recategorize transaction {txn_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to recategorize transaction"
        )

# Health check
@router.get("/health")
async def transactions_health_check():
    """Transactions service health check."""
    return {
        "status": "healthy",
        "service": "transactions",
        "timestamp": datetime.now().isoformat(),
        "categorizer_loaded": categorizer is not None
    }
