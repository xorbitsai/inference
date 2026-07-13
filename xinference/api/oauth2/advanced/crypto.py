# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import hmac
import logging
import os
import secrets
import string
from typing import Optional

import bcrypt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def aes_encrypt(plaintext: str, key: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError("AES key must be 32 bytes for AES-256-GCM")
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce + ciphertext


def aes_decrypt(ciphertext: bytes, key: bytes) -> Optional[str]:
    if len(key) != 32:
        raise ValueError("AES key must be 32 bytes for AES-256-GCM")
    nonce = ciphertext[:12]
    data = ciphertext[12:]
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, data, None)
        return plaintext.decode("utf-8")
    except Exception:
        logger.warning("AES-GCM decryption failed: invalid ciphertext or key")
        return None


def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    if len(pwd_bytes) > 72:
        pwd_bytes = hashlib.sha256(pwd_bytes).digest()[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    pwd_bytes = password.encode("utf-8")
    if len(pwd_bytes) > 72:
        pwd_bytes = hashlib.sha256(pwd_bytes).digest()[:72]
    return bcrypt.checkpw(pwd_bytes, hashed.encode("utf-8"))


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_api_key() -> str:
    chars = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(chars) for _ in range(48))
    return f"xf-{random_part}"


def derive_encryption_key(key_str: str) -> bytes:
    return hashlib.sha256(key_str.encode("utf-8")).digest()
