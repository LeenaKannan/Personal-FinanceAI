 # Bank statement parsing
import re
import PyPDF2
import pandas as pd

class PDFParser:
    def __init__(self):
        pass

    def extract_text(self, pdf_path):
        text = ''
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text

    def parse_transactions(self, text):
        transactions = []
        lines = text.split('\n')
        for line in lines:
            match = re.match(r'(\d{2}/\d{2}/\d{4})\s+(.*?)\s+(-?\d+\.?\d*)', line)
            if match:
                date, description, amount = match.groups()
                transactions.append({'date': date, 'description': description, 'amount': float(amount)})
        return pd.DataFrame(transactions)
