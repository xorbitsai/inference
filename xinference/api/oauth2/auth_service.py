# Copyright 2022-2024 XProbe Inc.
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
import re
from datetime import timedelta
from typing import List, Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from typing_extensions import Annotated

from ..._compat import BaseModel, ValidationError, parse_file_as
from .types import AuthStartupConfig, User
from .utils import create_access_token, get_password_hash, verify_password

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class TokenData(BaseModel):
    username: str
    scopes: List[str] = []


class AuthService:
    def __init__(self, auth_config_file: Optional[str]):
        self._auth_config_file = auth_config_file
        self._config = self.init_auth_config()

    @property
    def config(self):
        return self._config

    @staticmethod
    def is_legal_api_key(key: str) -> bool:
        pattern = re.compile("^sk-[a-zA-Z0-9]{13}$")
        return re.match(pattern, key) is not None

    def init_auth_config(self):
        if self._auth_config_file:
            config: AuthStartupConfig = parse_file_as(  # type: ignore
                path=self._auth_config_file, type_=AuthStartupConfig
            )
            all_api_keys = set()
            for user in config.user_config:
                user.password = get_password_hash(user.password)
                for api_key in user.api_keys:
                    if not self.is_legal_api_key(api_key):
                        raise ValueError(
                            "Api-Key should be a string started with 'sk-' with a total length of 16"
                        )
                    if api_key in all_api_keys:
                        raise ValueError(
                            "Duplicate api-keys exists, please check your configuration"
                        )
                    else:
                        all_api_keys.add(api_key)
            return config

    def __call__(
        self,
        security_scopes: SecurityScopes,
        token: Annotated[str, Depends(oauth2_scheme)],
    ):
        """
        Advanced dependencies. See: https://fastapi.tiangolo.com/advanced/advanced-dependencies/
        """
        if security_scopes.scopes:
            authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
        else:
            authenticate_value = "Bearer"
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )

        if self.is_legal_api_key(token):
            user, token_scopes = self.get_user_and_scopes_with_api_key(token)
        else:
            try:
                assert self._config is not None
                payload = jwt.decode(
                    token,
                    self._config.auth_config.secret_key,
                    algorithms=[self._config.auth_config.algorithm],
                    options={"verify_exp": False},  # TODO: supports token expiration
                )
                username: str = payload.get("sub")
                if username is None:
                    raise credentials_exception
                token_scopes = payload.get("scopes", [])
                user = self.get_user(username)
            except (JWTError, ValidationError):
                raise credentials_exception
        if user is None:
            raise credentials_exception
        if "admin" in token_scopes:
            return user
        for scope in security_scopes.scopes:
            if scope not in token_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions",
                    headers={"WWW-Authenticate": authenticate_value},
                )
        return user

    def get_user(self, username: str) -> Optional[User]:
        for user in self._config.user_config:
            if user.username == username:
                return user
        return None

    def get_user_and_scopes_with_api_key(
        self, api_key: str
    ) -> Tuple[Optional[User], List]:
        for user in self._config.user_config:
            for key in user.api_keys:
                if api_key == key:
                    return user, user.permissions
        return None, []

    def authenticate_user(self, username: str, password: str):
        user = self.get_user(username)
        if not user:
            return False
        if not verify_password(password, user.password):
            return False
        return user

    def generate_token_for_user(self, username: str, password: str):
        user = self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert user is not None and isinstance(user, User)
        access_token_expires = timedelta(
            minutes=self._config.auth_config.token_expire_in_minutes
        )
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.permissions},
            secret_key=self._config.auth_config.secret_key,
            algorithm=self._config.auth_config.algorithm,
            expires_delta=access_token_expires,
        )
        return {"access_token": access_token, "token_type": "bearer"}
