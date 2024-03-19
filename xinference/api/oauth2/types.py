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
from typing import List

from ..._compat import BaseModel


class LoginUserForm(BaseModel):
    username: str
    password: str


class User(LoginUserForm):
    permissions: List[str]
    api_keys: List[str]


class AuthConfig(BaseModel):
    algorithm: str = "HS256"
    secret_key: str
    token_expire_in_minutes: int


class AuthStartupConfig(BaseModel):
    auth_config: AuthConfig
    user_config: List[User]
