# src/mask_reg/hashing.py
import hashlib

def stable_hash_token(token: str, digest_size: int = 8) -> str:
    """
    Deterministic hash so we can store token stats without storing the raw token.
    digest_size=8 -> 16 hex chars (compact, enough for practice).
    """
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=digest_size)
    return h.hexdigest()
