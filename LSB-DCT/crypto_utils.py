from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

def kdf_lsb_mode(master_password, salt):
    """
    Derive a 32-byte Key only.
    Used by lsb_stego.py (Nonce is managed manually).
    """
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
    return kdf.derive(master_password.encode())

def kdf_fft_mode(master_password, salt):
    """
    Derive a 32-byte Key and a 12-byte Nonce using PBKDF2.
    Total length = 44 bytes.
    Used by freq_stego.py (Nonce is not stored to save space).
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=44,
        salt=salt,
        iterations=100000
    )
    derived = kdf.derive(master_password.encode())
    return derived[:32], derived[32:]

def encrypt_data(plaintext_bytes, key, nonce):
    """Encrypt and authenticate data using ChaCha20-Poly1305."""
    aead = ChaCha20Poly1305(key)
    return aead.encrypt(nonce, plaintext_bytes, None)  # Returns ciphertext + tag

def decrypt_metadata(encrypted_bytes_with_tag, key, nonce):
    """Verify and decrypt ChaCha20-Poly1305 metadata."""
    aead = ChaCha20Poly1305(key)
    try:
        return aead.decrypt(nonce, encrypted_bytes_with_tag, None)
    except InvalidTag:
        return None