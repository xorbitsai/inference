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
"""Offline command to reset an admin password in the advanced auth database.

This operates directly on the SQLite database and bypasses the API and its
permission checks, so it works even when nobody can log in (e.g. the only
admin password was lost, or someone else won the first-admin race). It only
touches the ``users`` and ``refresh_tokens`` tables, so no encryption key is
required.
"""

import argparse
import getpass
import logging
import os
import sys
from typing import Optional

from .crypto import get_password_hash
from .database import Database

logger = logging.getLogger(__name__)


def _password_min_length() -> int:
    """Resolve the minimum password length, mirroring the API's own rule
    (XINFERENCE_PASSWORD_MIN_LENGTH, default 8) so a reset can't set a
    password the login/change-password endpoints would later reject."""
    try:
        value = int(os.environ.get("XINFERENCE_PASSWORD_MIN_LENGTH", "8"))
        if value <= 0:
            raise ValueError("must be positive")
        return value
    except (TypeError, ValueError):
        logger.warning(
            "XINFERENCE_PASSWORD_MIN_LENGTH must be a positive integer (got %r); "
            "falling back to 8.",
            os.environ.get("XINFERENCE_PASSWORD_MIN_LENGTH"),
        )
        return 8


PASSWORD_MIN_LENGTH = _password_min_length()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="xinference-reset-auth-password",
        description="Reset the password for an admin user in the advanced auth database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xinference-reset-auth-password --username admin
  xinference-reset-auth-password --username admin --password "new-strong-password"
  xinference-reset-auth-password --yes
        """,
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Admin username to reset (if omitted, will prompt; default: admin)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="New password (if omitted, will prompt securely)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the auth SQLite database "
        "(default: XINFERENCE_AUTH_DB_PATH, i.e. <XINFERENCE_HOME>/auth/auth.db)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    return parser.parse_args()


def _read_username(username_from_args: Optional[str]) -> str:
    if username_from_args and username_from_args.strip():
        return username_from_args.strip()

    entered = input("Admin username [admin]: ").strip()
    return entered or "admin"


def _read_password(password_from_args: Optional[str]) -> str:
    if password_from_args:
        if len(password_from_args) < PASSWORD_MIN_LENGTH:
            raise ValueError(
                f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
            )
        return password_from_args

    print(f"New password must be at least {PASSWORD_MIN_LENGTH} characters.")
    while True:
        password = getpass.getpass("New password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match. Please try again.")
            continue
        if len(password) < PASSWORD_MIN_LENGTH:
            print(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")
            continue
        return password


def _confirm_reset(username: str, skip_confirmation: bool) -> None:
    if skip_confirmation:
        return

    confirmed = input(
        f"Reset password for admin '{username}' and revoke active refresh tokens? [y/N]: "
    ).strip()
    if confirmed.lower() not in {"y", "yes"}:
        raise ValueError("Operation cancelled")


def reset_admin_password(db: Database, username: str, new_password: str) -> None:
    if len(new_password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")

    user = db.get_user_by_username(username, source="local")
    if user is None:
        raise ValueError(f"User '{username}' not found")
    if "admin" not in (user.get("permissions") or []):
        raise ValueError(f"User '{username}' is not an admin")

    # Atomically update the password and revoke the user's refresh tokens, so
    # a leaked/old refresh token can't outlive the reset.
    db.update_password_and_revoke_tokens(user["id"], get_password_hash(new_password))


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    try:
        if args.db_path:
            db_path = args.db_path
        else:
            from ....constants import XINFERENCE_AUTH_DB_PATH

            db_path = XINFERENCE_AUTH_DB_PATH

        db = Database(db_path)
        username = _read_username(args.username)
        _confirm_reset(username, args.yes)
        new_password = _read_password(args.password)
        reset_admin_password(db, username, new_password)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to reset admin password: {exc}")
        sys.exit(1)

    print(f"Password for admin '{username}' has been reset successfully.")


if __name__ == "__main__":
    main()
