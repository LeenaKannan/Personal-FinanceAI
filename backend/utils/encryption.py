# backend/utils/encryption.py

import os
import base64
import hashlib
import secrets
from typing import Union, Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EncryptionUtils:
    """
    Comprehensive encryption utilities for Personal Finance AI application.
    Supports symmetric encryption, asymmetric encryption, password hashing, and secure key management.
    """
    
    def __init__(self, key: Optional[bytes] = None, key_file: Optional[str] = None):
        """
        Initialize encryption utilities.
        
        Args:
            key: Optional encryption key (generates new if None)
            key_file: Optional path to key file
        """
        if key_file and os.path.exists(key_file):
            self.key = self._load_key_from_file(key_file)
        elif key:
            self.key = key
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        self.backend = default_backend()

    # === SYMMETRIC ENCRYPTION (Fernet) ===
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using Fernet symmetric encryption.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            bytes: Encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            return self.cipher.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt(self, token: bytes) -> bytes:
        """
        Decrypt data using Fernet symmetric encryption.
        
        Args:
            token: Encrypted data to decrypt
            
        Returns:
            bytes: Decrypted data
        """
        try:
            return self.cipher.decrypt(token)
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise

    def encrypt_string(self, text: str) -> str:
        """
        Encrypt string and return base64 encoded result.
        
        Args:
            text: String to encrypt
            
        Returns:
            str: Base64 encoded encrypted string
        """
        encrypted_bytes = self.encrypt(text)
        return base64.b64encode(encrypted_bytes).decode('utf-8')

    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt base64 encoded string.
        
        Args:
            encrypted_text: Base64 encoded encrypted string
            
        Returns:
            str: Decrypted string
        """
        encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted_bytes = self.decrypt(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')

    def encrypt_json(self, data: Dict[str, Any]) -> str:
        """
        Encrypt JSON data.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            str: Base64 encoded encrypted JSON
        """
        json_string = json.dumps(data, separators=(',', ':'))
        return self.encrypt_string(json_string)

    def decrypt_json(self, encrypted_json: str) -> Dict[str, Any]:
        """
        Decrypt JSON data.
        
        Args:
            encrypted_json: Base64 encoded encrypted JSON
            
        Returns:
            dict: Decrypted dictionary
        """
        json_string = self.decrypt_string(encrypted_json)
        return json.loads(json_string)

    # === ADVANCED SYMMETRIC ENCRYPTION (AES) ===
    
    def encrypt_aes_gcm(self, data: Union[str, bytes], key: Optional[bytes] = None) -> Dict[str, str]:
        """
        Encrypt data using AES-GCM (provides authentication).
        
        Args:
            data: Data to encrypt
            key: Optional 32-byte key (generates if None)
            
        Returns:
            dict: Contains encrypted data, nonce, and tag (all base64 encoded)
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if not key:
                key = os.urandom(32)  # 256-bit key
            
            nonce = os.urandom(12)  # 96-bit nonce for GCM
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            return {
                'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
                'nonce': base64.b64encode(nonce).decode('utf-8'),
                'tag': base64.b64encode(encryptor.tag).decode('utf-8'),
                'key': base64.b64encode(key).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"AES-GCM encryption failed: {str(e)}")
            raise

    def decrypt_aes_gcm(self, encrypted_data: Dict[str, str]) -> bytes:
        """
        Decrypt AES-GCM encrypted data.
        
        Args:
            encrypted_data: Dictionary with ciphertext, nonce, tag, and key
            
        Returns:
            bytes: Decrypted data
        """
        try:
            ciphertext = base64.b64decode(encrypted_data['ciphertext'])
            nonce = base64.b64decode(encrypted_data['nonce'])
            tag = base64.b64decode(encrypted_data['tag'])
            key = base64.b64decode(encrypted_data['key'])
            
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend)
            decryptor = cipher.decryptor()
            
            return decryptor.update(ciphertext) + decryptor.finalize()
        except Exception as e:
            logger.error(f"AES-GCM decryption failed: {str(e)}")
            raise

    # === ASYMMETRIC ENCRYPTION (RSA) ===
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Dict[str, bytes]:
        """
        Generate RSA public/private key pair.
        
        Args:
            key_size: Key size in bits (default: 2048)
            
        Returns:
            dict: Contains 'private_key' and 'public_key' in PEM format
        """
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=self.backend
            )
            
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return {
                'private_key': private_pem,
                'public_key': public_pem
            }
        except Exception as e:
            logger.error(f"RSA key generation failed: {str(e)}")
            raise

    def encrypt_rsa(self, data: Union[str, bytes], public_key_pem: bytes) -> bytes:
        """
        Encrypt data using RSA public key.
        
        Args:
            data: Data to encrypt
            public_key_pem: Public key in PEM format
            
        Returns:
            bytes: Encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            public_key = serialization.load_pem_public_key(public_key_pem, backend=self.backend)
            
            encrypted = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return encrypted
        except Exception as e:
            logger.error(f"RSA encryption failed: {str(e)}")
            raise

    def decrypt_rsa(self, encrypted_data: bytes, private_key_pem: bytes) -> bytes:
        """
        Decrypt RSA encrypted data.
        
        Args:
            encrypted_data: Encrypted data
            private_key_pem: Private key in PEM format
            
        Returns:
            bytes: Decrypted data
        """
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem, 
                password=None, 
                backend=self.backend
            )
            
            decrypted = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted
        except Exception as e:
            logger.error(f"RSA decryption failed: {str(e)}")
            raise

    # === PASSWORD HASHING ===
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Hash password using Scrypt (recommended for passwords).
        
        Args:
            password: Password to hash
            salt: Optional salt (generates if None)
            
        Returns:
            dict: Contains 'hash' and 'salt' (both base64 encoded)
        """
        try:
            if not salt:
                salt = os.urandom(32)
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,  # CPU/memory cost parameter
                r=8,      # Block size parameter
                p=1,      # Parallelization parameter
                backend=self.backend
            )
            
            key = kdf.derive(password.encode('utf-8'))
            
            return {
                'hash': base64.b64encode(key).decode('utf-8'),
                'salt': base64.b64encode(salt).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Password hashing failed: {str(e)}")
            raise

    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            password: Password to verify
            stored_hash: Base64 encoded stored hash
            salt: Base64 encoded salt
            
        Returns:
            bool: True if password matches, False otherwise
        """
        try:
            salt_bytes = base64.b64decode(salt)
            stored_hash_bytes = base64.b64decode(stored_hash)
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                n=2**14,
                r=8,
                p=1,
                backend=self.backend
            )
            
            try:
                kdf.verify(password.encode('utf-8'), stored_hash_bytes)
                return True
            except:
                return False
        except Exception as e:
            logger.error(f"Password verification failed: {str(e)}")
            return False

    # === KEY DERIVATION ===
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Optional salt (generates if None)
            
        Returns:
            dict: Contains 'key' and 'salt' (both base64 encoded)
        """
        try:
            if not salt:
                salt = os.urandom(32)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,  # Recommended minimum
                backend=self.backend
            )
            
            key = kdf.derive(password.encode('utf-8'))
            
            return {
                'key': base64.b64encode(key).decode('utf-8'),
                'salt': base64.b64encode(salt).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Key derivation failed: {str(e)}")
            raise

    # === SECURE RANDOM GENERATION ===
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate cryptographically secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            str: Base64 encoded secure token
        """
        token = secrets.token_bytes(length)
        return base64.b64encode(token).decode('utf-8')

    def generate_api_key(self, prefix: str = "sk") -> str:
        """
        Generate API key with prefix.
        
        Args:
            prefix: Key prefix (default: "sk")
            
        Returns:
            str: API key in format "prefix_base64token"
        """
        token = self.generate_secure_token(24)
        return f"{prefix}_{token}"

    # === KEY MANAGEMENT ===
    
    def get_key(self) -> bytes:
        """Get the current encryption key."""
        return self.key

    def save_key_to_file(self, file_path: str) -> None:
        """
        Save encryption key to file securely.
        
        Args:
            file_path: Path to save key file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(self.key)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(file_path, 0o600)
            logger.info(f"Key saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save key: {str(e)}")
            raise

    def _load_key_from_file(self, file_path: str) -> bytes:
        """Load encryption key from file."""
        try:
            with open(file_path, 'rb') as f:
                key = f.read()
            logger.info(f"Key loaded from {file_path}")
            return key
        except Exception as e:
            logger.error(f"Failed to load key: {str(e)}")
            raise

    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """
        Rotate encryption key.
        
        Args:
            new_key: Optional new key (generates if None)
            
        Returns:
            bytes: New encryption key
        """
        old_key = self.key
        
        if not new_key:
            new_key = Fernet.generate_key()
        
        self.key = new_key
        self.cipher = Fernet(self.key)
        
        logger.info("Encryption key rotated")
        return new_key

    # === FINANCIAL DATA SPECIFIC ENCRYPTION ===
    
    def encrypt_financial_data(self, data: Dict[str, Any]) -> str:
        """
        Encrypt financial data with additional security measures.
        
        Args:
            data: Financial data dictionary
            
        Returns:
            str: Encrypted financial data
        """
        try:
            # Add timestamp and checksum for integrity
            data['_timestamp'] = int(secrets.randbits(64))
            data['_checksum'] = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            
            return self.encrypt_json(data)
        except Exception as e:
            logger.error(f"Financial data encryption failed: {str(e)}")
            raise

    def decrypt_financial_data(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt and verify financial data.
        
        Args:
            encrypted_data: Encrypted financial data
            
        Returns:
            dict: Decrypted and verified financial data
        """
        try:
            data = self.decrypt_json(encrypted_data)
            
            # Verify checksum
            stored_checksum = data.pop('_checksum', None)
            data.pop('_timestamp', None)  # Remove timestamp
            
            calculated_checksum = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            
            if stored_checksum != calculated_checksum:
                raise ValueError("Data integrity check failed")
            
            return data
        except Exception as e:
            logger.error(f"Financial data decryption failed: {str(e)}")
            raise

    # === UTILITY METHODS ===
    
    def secure_compare(self, a: str, b: str) -> bool:
        """
        Constant-time string comparison to prevent timing attacks.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            bool: True if strings are equal
        """
        return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

# Example usage and testing
if __name__ == "__main__":
    # Test the encryption utilities
    print("Testing EncryptionUtils...")
    
    # Initialize encryption
    crypto = EncryptionUtils()
    
    # Test symmetric encryption
    original_text = "Sensitive financial data: Account balance ₹1,50,000"
    encrypted = crypto.encrypt_string(original_text)
    decrypted = crypto.decrypt_string(encrypted)
    print(f"Symmetric encryption test: {original_text == decrypted}")
    
    # Test JSON encryption
    financial_data = {
        "account_number": "123456789",
        "balance": 150000.50,
        "transactions": [
            {"date": "2025-06-23", "amount": -850, "description": "ZOMATO"}
        ]
    }
    
    encrypted_json = crypto.encrypt_financial_data(financial_data)
    decrypted_json = crypto.decrypt_financial_data(encrypted_json)
    print(f"Financial data encryption test: {financial_data == decrypted_json}")
    
    # Test password hashing
    password = "MySecurePassword123!"
    hash_result = crypto.hash_password(password)
    verification = crypto.verify_password(password, hash_result['hash'], hash_result['salt'])
    print(f"Password hashing test: {verification}")
    
    # Test RSA encryption
    keypair = crypto.generate_rsa_keypair()
    rsa_data = "RSA test data"
    rsa_encrypted = crypto.encrypt_rsa(rsa_data, keypair['public_key'])
    rsa_decrypted = crypto.decrypt_rsa(rsa_encrypted, keypair['private_key'])
    print(f"RSA encryption test: {rsa_data == rsa_decrypted.decode('utf-8')}")
    
    # Test secure token generation
    api_key = crypto.generate_api_key("finai")
    print(f"Generated API key: {api_key}")
    
    print("All encryption tests completed!")
