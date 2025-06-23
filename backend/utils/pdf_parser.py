# backend/utils/pdf_parser.py

import re
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from decimal import Decimal, InvalidOperation
import io

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import tabula
except ImportError:
    tabula = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFParser:
    """
    Comprehensive PDF parser for Indian bank statement transactions.
    Supports multiple banks (ICICI, HDFC, SBI, Axis, Kotak, etc.) with fallback parsing methods.
    """

    def __init__(self):
        """Initialize PDF parser with bank-specific patterns."""
        self.bank_patterns = self._initialize_bank_patterns()
        self.date_patterns = self._initialize_date_patterns()
        self.amount_patterns = self._initialize_amount_patterns()

    def _initialize_bank_patterns(self) -> Dict[str, Dict]:
        """Initialize regex patterns for different Indian banks."""
        return {
            'ICICI': {
                'transaction_pattern': r'(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
                'balance_pattern': r'Balance\s*:\s*Rs\.?\s*([\d,]+\.?\d*)',
                'account_pattern': r'Account\s*No\.?\s*:?\s*(\d+)',
                'date_format': '%d/%m/%Y'
            },
            'HDFC': {
                'transaction_pattern': r'(\d{2}/\d{2}/\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s*([CD])\s+([\d,]+\.?\d*)',
                'balance_pattern': r'Balance\s*Rs\.?\s*([\d,]+\.?\d*)',
                'account_pattern': r'A/c\s*No\.?\s*:?\s*(\d+)',
                'date_format': '%d/%m/%Y'
            },
            'SBI': {
                'transaction_pattern': r'(\d{2}-\d{2}-\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
                'balance_pattern': r'Balance\s*:\s*Rs\.?\s*([\d,]+\.?\d*)',
                'account_pattern': r'Account\s*Number\s*:?\s*(\d+)',
                'date_format': '%d-%m-%Y'
            },
            'AXIS': {
                'transaction_pattern': r'(\d{2}/\d{2}/\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
                'balance_pattern': r'Balance\s*:\s*Rs\.?\s*([\d,]+\.?\d*)',
                'account_pattern': r'Account\s*No\.?\s*:?\s*(\d+)',
                'date_format': '%d/%m/%Y'
            },
            'KOTAK': {
                'transaction_pattern': r'(\d{2}/\d{2}/\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
                'balance_pattern': r'Balance\s*:\s*Rs\.?\s*([\d,]+\.?\d*)',
                'account_pattern': r'Account\s*Number\s*:?\s*(\d+)',
                'date_format': '%d/%m/%Y'
            }
        }

    def _initialize_date_patterns(self) -> List[str]:
        """Initialize common date patterns found in Indian bank statements."""
        return [
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{2}\.\d{2}\.\d{4}', # DD.MM.YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        ]

    def _initialize_amount_patterns(self) -> List[str]:
        """Initialize amount patterns for Indian currency formatting."""
        return [
            r'Rs\.?\s*([\d,]+\.?\d*)',  # Rs. 1,000.00
            r'₹\s*([\d,]+\.?\d*)',      # ₹ 1,000.00
            r'([\d,]+\.?\d*)\s*Rs\.?',  # 1,000.00 Rs.
            r'([\d,]+\.?\d*)\s*INR',    # 1,000.00 INR
            r'([\d,]+\.?\d*)',          # Plain number
        ]

    def extract_text(self, pdf_path: str, method: str = 'auto') -> str:
        """
        Extract text from PDF using multiple methods with fallback.
        
        Args:
            pdf_path: Path to the PDF file
            method: Extraction method ('pypdf2', 'pdfplumber', 'tabula', 'auto')
            
        Returns:
            str: Extracted text from PDF
        """
        if method == 'auto':
            # Try methods in order of preference
            for extraction_method in ['pdfplumber', 'pypdf2', 'tabula']:
                try:
                    text = self._extract_with_method(pdf_path, extraction_method)
                    if text and len(text.strip()) > 100:  # Minimum content check
                        logger.info(f"Successfully extracted text using {extraction_method}")
                        return text
                except Exception as e:
                    logger.warning(f"Failed to extract with {extraction_method}: {str(e)}")
                    continue
            
            raise Exception("All extraction methods failed")
        else:
            return self._extract_with_method(pdf_path, method)

    def _extract_with_method(self, pdf_path: str, method: str) -> str:
        """Extract text using specific method."""
        if method == 'pypdf2':
            return self._extract_with_pypdf2(pdf_path)
        elif method == 'pdfplumber':
            return self._extract_with_pdfplumber(pdf_path)
        elif method == 'tabula':
            return self._extract_with_tabula(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        if not PyPDF2:
            raise ImportError("PyPDF2 not installed")
        
        text = ''
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            raise
        
        return text

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for tables)."""
        if not pdfplumber:
            raise ImportError("pdfplumber not installed")
        
        text = ''
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        for table_num, table in enumerate(tables):
                            text += f"\n--- Table {table_num + 1} on Page {page_num + 1} ---\n"
                            for row in table:
                                if row:
                                    text += '\t'.join([cell or '' for cell in row]) + '\n'
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            raise
        
        return text

    def _extract_with_tabula(self, pdf_path: str) -> str:
        """Extract tables using tabula-py."""
        if not tabula:
            raise ImportError("tabula-py not installed")
        
        try:
            # Extract all tables from PDF
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            text = ''
            
            for table_num, df in enumerate(tables):
                text += f"\n--- Table {table_num + 1} ---\n"
                text += df.to_string(index=False) + '\n'
            
            return text
        except Exception as e:
            logger.error(f"tabula extraction failed: {str(e)}")
            raise

    def detect_bank(self, text: str) -> Optional[str]:
        """
        Detect bank from statement text.
        
        Args:
            text: Extracted text from bank statement
            
        Returns:
            str: Detected bank name or None
        """
        text_upper = text.upper()
        
        bank_indicators = {
            'ICICI': ['ICICI BANK', 'ICICI', 'ICICIBANK'],
            'HDFC': ['HDFC BANK', 'HDFC', 'HDFCBANK'],
            'SBI': ['STATE BANK OF INDIA', 'SBI', 'SBIN'],
            'AXIS': ['AXIS BANK', 'AXIS', 'AXISBANK'],
            'KOTAK': ['KOTAK MAHINDRA BANK', 'KOTAK', 'KOTAKBANK'],
            'PNB': ['PUNJAB NATIONAL BANK', 'PNB'],
            'BOI': ['BANK OF INDIA', 'BOI'],
            'CANARA': ['CANARA BANK', 'CANARA'],
            'UNION': ['UNION BANK', 'UNION'],
            'IDBI': ['IDBI BANK', 'IDBI']
        }
        
        for bank, indicators in bank_indicators.items():
            for indicator in indicators:
                if indicator in text_upper:
                    logger.info(f"Detected bank: {bank}")
                    return bank
        
        logger.warning("Could not detect bank from statement")
        return None

    def parse_transactions(self, text: str, bank: Optional[str] = None) -> pd.DataFrame:
        """
        Parse transactions from extracted text.
        
        Args:
            text: Extracted text from bank statement
            bank: Bank name (auto-detected if None)
            
        Returns:
            pd.DataFrame: Parsed transactions
        """
        if not bank:
            bank = self.detect_bank(text)
        
        # Try bank-specific parsing first
        if bank and bank in self.bank_patterns:
            try:
                transactions = self._parse_bank_specific(text, bank)
                if transactions:
                    logger.info(f"Successfully parsed {len(transactions)} transactions using {bank} pattern")
                    return pd.DataFrame(transactions)
            except Exception as e:
                logger.warning(f"Bank-specific parsing failed for {bank}: {str(e)}")
        
        # Fallback to generic parsing
        logger.info("Using generic parsing method")
        transactions = self._parse_generic(text)
        return pd.DataFrame(transactions)

    def _parse_bank_specific(self, text: str, bank: str) -> List[Dict]:
        """Parse transactions using bank-specific patterns."""
        patterns = self.bank_patterns[bank]
        transactions = []
        
        lines = text.split('\n')
        transaction_pattern = re.compile(patterns['transaction_pattern'])
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = transaction_pattern.search(line)
            if match:
                try:
                    transaction = self._extract_transaction_data(match, patterns['date_format'])
                    if transaction:
                        transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Failed to parse line: {line[:50]}... Error: {str(e)}")
                    continue
        
        return transactions

    def _parse_generic(self, text: str) -> List[Dict]:
        """Generic transaction parsing with multiple patterns."""
        transactions = []
        lines = text.split('\n')
        
        # Generic patterns for different transaction formats
        generic_patterns = [
            # Pattern 1: Date Description Amount Balance
            r'(\d{2}[/-]\d{2}[/-]\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
            # Pattern 2: Date Date Description Debit Credit Balance
            r'(\d{2}[/-]\d{2}[/-]\d{4})\s+(\d{2}[/-]\d{2}[/-]\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
            # Pattern 3: Date Description Amount (with Dr/Cr indicator)
            r'(\d{2}[/-]\d{2}[/-]\d{4})\s+(.*?)\s+([\d,]+\.?\d*)\s*([CD]r?)',
            # Pattern 4: Simple date description amount
            r'(\d{2}[/-]\d{2}[/-]\d{4})\s+(.*?)\s+(-?[\d,]+\.?\d*)',
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            for pattern in generic_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        transaction = self._extract_generic_transaction(match)
                        if transaction:
                            transactions.append(transaction)
                            break  # Found a match, move to next line
                    except Exception as e:
                        logger.warning(f"Failed to parse line: {line[:50]}... Error: {str(e)}")
                        continue
        
        return transactions

    def _extract_transaction_data(self, match: re.Match, date_format: str) -> Optional[Dict]:
        """Extract transaction data from regex match."""
        groups = match.groups()
        
        if len(groups) < 3:
            return None
        
        try:
            # Parse date
            date_str = groups[0]
            date_obj = datetime.strptime(date_str, date_format)
            normalized_date = date_obj.strftime('%Y-%m-%d')
            
            # Extract description (usually the longest text group)
            description = groups[1] if len(groups) > 1 else "Unknown Transaction"
            description = self._clean_description(description)
            
            # Extract amount (try different groups)
            amount = 0.0
            for i in range(2, len(groups)):
                try:
                    amount_str = groups[i]
                    amount = self._parse_amount(amount_str)
                    if amount != 0:
                        break
                except:
                    continue
            
            return {
                'date': normalized_date,
                'description': description,
                'amount': amount,
                'raw_line': match.string
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract transaction data: {str(e)}")
            return None

    def _extract_generic_transaction(self, match: re.Match) -> Optional[Dict]:
        """Extract transaction from generic pattern match."""
        groups = match.groups()
        
        try:
            # Parse date (first group is always date)
            date_str = groups[0]
            date_obj = self._parse_date(date_str)
            normalized_date = date_obj.strftime('%Y-%m-%d')
            
            # Find description (longest non-numeric group)
            description = ""
            amount = 0.0
            
            for group in groups[1:]:
                if self._is_amount(group):
                    if amount == 0:  # Take first amount found
                        amount = self._parse_amount(group)
                else:
                    if len(group) > len(description):
                        description = group
            
            description = self._clean_description(description)
            
            return {
                'date': normalized_date,
                'description': description,
                'amount': amount,
                'raw_line': match.string
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract generic transaction: {str(e)}")
            return None

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string with multiple format attempts."""
        date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%Y-%m-%d']
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_str}")

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float."""
        if not amount_str:
            return 0.0
        
        # Remove currency symbols and spaces
        amount_clean = re.sub(r'[Rs₹\s,]', '', amount_str)
        amount_clean = amount_clean.replace('Dr', '').replace('Cr', '').strip()
        
        try:
            return float(amount_clean)
        except ValueError:
            return 0.0

    def _is_amount(self, text: str) -> bool:
        """Check if text represents an amount."""
        if not text:
            return False
        
        # Remove common currency indicators
        clean_text = re.sub(r'[Rs₹\s,DrCr]', '', text).strip()
        
        try:
            float(clean_text)
            return True
        except ValueError:
            return False

    def _clean_description(self, description: str) -> str:
        """Clean and normalize transaction description."""
        if not description:
            return "Unknown Transaction"
        
        # Remove extra spaces and special characters
        description = re.sub(r'\s+', ' ', description).strip()
        description = re.sub(r'[^\w\s\-\.]', ' ', description)
        
        # Limit length
        return description[:200]

    def extract_account_info(self, text: str) -> Dict[str, str]:
        """Extract account information from statement."""
        info = {}
        
        # Account number patterns
        account_patterns = [
            r'Account\s*(?:No\.?|Number)\s*:?\s*(\d+)',
            r'A/c\s*No\.?\s*:?\s*(\d+)',
            r'Account\s*:?\s*(\d+)',
        ]
        
        for pattern in account_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['account_number'] = match.group(1)
                break
        
        # IFSC code
        ifsc_match = re.search(r'IFSC\s*:?\s*([A-Z]{4}0[A-Z0-9]{6})', text, re.IGNORECASE)
        if ifsc_match:
            info['ifsc_code'] = ifsc_match.group(1)
        
        # Statement period
        period_match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})\s*(?:to|TO|-)\s*(\d{2}[/-]\d{2}[/-]\d{4})', text)
        if period_match:
            info['statement_from'] = period_match.group(1)
            info['statement_to'] = period_match.group(2)
        
        return info

    def validate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean parsed transactions."""
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'description', 'amount'])
        
        # Filter out invalid dates
        df = df[pd.to_datetime(df['date'], errors='coerce').notna()]
        
        # Filter out zero amounts
        df = df[df['amount'] != 0]
        
        # Sort by date
        df = df.sort_values('date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df

# Example usage and testing
if __name__ == "__main__":
    # Test the parser
    parser = PDFParser()
    
    # Example with mock data
    sample_text = """
    ICICI BANK LIMITED
    Account No: 123456789012
    IFSC: ICIC0001234
    
    01/06/2025  UPI-ZOMATO ONLINE PAYMENT  850.00  45,150.00
    02/06/2025  RENT TRANSFER TO LANDLORD  28,000.00  17,150.00
    03/06/2025  SALARY CREDIT  80,000.00  97,150.00
    """
    
    # Parse transactions
    transactions_df = parser.parse_transactions(sample_text)
    print("Parsed Transactions:")
    print(transactions_df)
    
    # Extract account info
    account_info = parser.extract_account_info(sample_text)
    print("\nAccount Information:")
    print(account_info)
