# Copyright 2022-2023 XProbe Inc.
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
import logging
from typing import List, Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError
from typing_extensions import Annotated

from .types import AuthStartupConfig, User

logger = logging.getLogger(__name__)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_db():
    from .common import XINFERENCE_OAUTH2_CONFIG

    # In a real enterprise-level environment, this should be the database
    yield XINFERENCE_OAUTH2_CONFIG


def get_user(db_users: List[User], username: str) -> Optional[User]:
    for user in db_users:
        if user.username == username:
            return user
    return None


class TokenData(BaseModel):
    username: Union[str, None] = None
    scopes: List[str] = []


def verify_token(
    security_scopes: SecurityScopes,
    token: Annotated[str, Depends(oauth2_scheme)],
    config: Optional[AuthStartupConfig] = Depends(get_db),
):
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        assert config is not None
        payload = jwt.decode(
            token,
            config.auth_config.secret_key,
            algorithms=[config.auth_config.algorithm],
            options={"verify_exp": False},  # TODO: supports token expiration
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        # TODO: check expire
        token_data = TokenData(scopes=token_scopes, username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    user = get_user(config.user_config, username=token_data.username)  # type: ignore
    if user is None:
        raise credentials_exception
    if "admin" in token_data.scopes:
        return user
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return user
