 # Security utilities
from cryptography.fernet import Fernet

class EncryptionUtils:
    def __init__(self, key=None):
        if key is None:
            key = Fernet.generate_key()
        self.key = key
        self.cipher = Fernet(self.key)

    def encrypt(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        return self.cipher.decrypt(token)

    def get_key(self) -> bytes:
        return self.key
