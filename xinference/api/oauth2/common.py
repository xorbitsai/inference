# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "lichengjie": {
        "username": "lichengjie",
        "full_name": "lichengjie",
        "email": "lichengjie@xprobe.io",
        "hashed_password": "$2b$12$VyoOP0H1gg42R.Raw4aPiOseVJS/RZNASluUEZdRpObV8iqvsWd96",
        "disabled": False,
    },
    "hekaisheng": {
        "username": "hekaisheng",
        "full_name": "hekaisheng",
        "email": "hekaisheng@xprobe.io",
        "hashed_password": "$2b$12$T7b13uszQVXey9tk9JI.d.hwuTbB/cNszBC/7wk5RjgIg/dC..neK",
        "disabled": True,
    },
}
