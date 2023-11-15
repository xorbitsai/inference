import logging
from typing import Union

from jose import JWTError, jwt
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .common import ALGORITHM, SECRET_KEY, fake_users_db

logger = logging.getLogger(__name__)


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


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


def error_response(error_msg: str, status_code: int) -> JSONResponse:
    logger.error(error_msg)
    return JSONResponse(
        content={"detail": error_msg},
        status_code=status_code,
    )


class OAuth2Middleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        logger.debug(f"Request URL path: {request.url.path}")
        if not request.url.path.startswith("/token"):
            if "Authorization" not in request.headers:
                return error_response("Could not validate credentials", 401)
            try:
                token_header = request.headers["Authorization"]
                if token_header.startswith("Bearer "):
                    token = token_header.split("Bearer ")[-1]
                else:
                    return error_response("Token should begin with `Bearer`", 401)

                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                username: str = payload.get("sub")
                if username is None:
                    return error_response("Invalid Bearer Token", 401)
                # TODO: check expire
                token_data = TokenData(username=username)
            except JWTError:
                return error_response("Invalid Bearer Token", 401)
            user = get_user(fake_users_db, username=token_data.username)  # type: ignore
            if user is None:
                return error_response("Invalid Bearer Token", 401)
            if user.disabled:
                return error_response("Inactive user", 403)
        response = await call_next(request)
        return response
