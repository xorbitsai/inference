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
"""Migration script: old auth config JSON → selected advanced auth store."""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

from .auth_store import create_auth_store
from .crypto import aes_encrypt, derive_encryption_key, get_password_hash, sha256_hex
from .database import ApiKeyNameConflictError

logger = logging.getLogger(__name__)


def migrate(
    auth_config_path: str,
    db_path: str,
    encryption_key: str,
    force: bool = False,
    dry_run: bool = False,
):
    config_file = Path(auth_config_path)
    if not config_file.exists():
        print(f"ERROR: Config file not found: {auth_config_path}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    users_config = config.get("user_config", [])
    print(f"Found {len(users_config)} users in old config")

    if dry_run:
        print("\n--- DRY RUN ---")
        for u in users_config:
            print(
                f"  User: {u['username']} | permissions: {u.get('permissions', [])} | keys: {len(u.get('api_keys', []))}"
            )
        print("--- END DRY RUN ---")
        return

    db = create_auth_store(db_path, purpose="migrate")
    if db.user_count() > 0 and not force:
        print("ERROR: Target database already has users. Use --force to overwrite.")
        sys.exit(1)

    enc_key = derive_encryption_key(encryption_key)

    for u in users_config:
        username = u["username"]
        password = u["password"]
        permissions = u.get("permissions", [])
        api_keys = u.get("api_keys", [])

        password_hash = get_password_hash(password)
        user_id = db.create_user(
            username=username,
            password_hash=password_hash,
            source="local",
            permissions=permissions,
        )
        print(f"  Migrated user: {username} (id={user_id})")

        for key_plaintext in api_keys:
            key_hash = sha256_hex(key_plaintext)
            key_encrypted = base64.b64encode(
                aes_encrypt(key_plaintext, enc_key)
            ).decode("utf-8")
            key_prefix = key_plaintext[:7]
            base_name = f"migrated-{key_prefix}"
            name = base_name
            suffix = 2
            while True:
                try:
                    db.create_api_key(
                        user_id=user_id,
                        key_hash=key_hash,
                        key_encrypted=key_encrypted,
                        key_prefix=key_prefix,
                        name=name,
                        model_permissions=[
                            {"permission_type": "all", "permission_value": None}
                        ],
                    )
                    break
                except ApiKeyNameConflictError:
                    name = f"{base_name}-{suffix}"
                    suffix += 1
            print(f"    Migrated key: {key_prefix}...")

    destination = f" to {db.path}" if db.path is not None else ""
    print(f"\nMigration complete. {len(users_config)} users migrated{destination}")


def main():
    parser = argparse.ArgumentParser(
        prog="xinference-migrate-auth",
        description="Migrate old xinference auth config to the selected auth store",
    )
    parser.add_argument(
        "--auth-config", required=True, help="Path to old auth config JSON"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Default SQLite path passed to the selected auth store provider",
    )
    parser.add_argument(
        "--encryption-key", required=True, help="Encryption key for API keys"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )

    args = parser.parse_args()
    migrate(
        auth_config_path=args.auth_config,
        db_path=args.db_path,
        encryption_key=args.encryption_key,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
