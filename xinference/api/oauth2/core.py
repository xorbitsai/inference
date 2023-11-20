import logging
from typing import Annotated, List, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError
from starlette.responses import JSONResponse

from .common import ALGORITHM, SECRET_KEY, fake_users_db

logger = logging.getLogger(__name__)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={"me": "Read information about the current user.", "items": "Read items."},
)


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    permissions: Union[List[str], None] = None


class UserInDB(User):
    hashed_password: str


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None
    scopes: list[str] = []


def error_response(error_msg: str, status_code: int) -> JSONResponse:
    logger.error(error_msg)
    return JSONResponse(
        content={"detail": error_msg},
        status_code=status_code,
    )


def verify_token(
    security_scopes: SecurityScopes, token: Annotated[str, Depends(oauth2_scheme)]
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
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        # TODO: check expire
        token_data = TokenData(scopes=token_scopes, username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)  # type: ignore
    if user is None:
        raise credentials_exception
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return user


# class OAuth2Middleware(BaseHTTPMiddleware):
#     async def dispatch(
#         self, request: Request, call_next: RequestResponseEndpoint
#     ) -> Response:
#         logger.debug(f"Request URL path: {request.url.path}")
#         if not request.url.path.startswith("/token"):
#             if "Authorization" not in request.headers:
#                 return error_response("Could not validate credentials", 401)
#             try:
#                 token_header = request.headers["Authorization"]
#                 if token_header.startswith("Bearer "):
#                     token = token_header.split("Bearer ")[-1]
#                 else:
#                     return error_response("Token should begin with `Bearer`", 401)
#
#                 payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#                 username: str = payload.get("sub")
#                 if username is None:
#                     return error_response("Invalid Bearer Token", 401)
#                 # TODO: check expire
#                 token_data = TokenData(username=username)
#             except JWTError:
#                 return error_response("Invalid Bearer Token", 401)
#             user = get_user(fake_users_db, username=token_data.username)  # type: ignore
#             if user is None:
#                 return error_response("Invalid Bearer Token", 401)
#             if user.disabled:
#                 return error_response("Inactive user", 403)
#         response = await call_next(request)
#         return response
