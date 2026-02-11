# from datetime import datetime, timedelta
# import os
# from dotenv import load_dotenv
# from argon2 import PasswordHasher
# from fastapi import HTTPException
# import jwt

# from core.config import settings

# ph = PasswordHasher()
# # from fastapi.security import OAuth2PasswordBearer


# load_dotenv()

# # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# def create_access_token(data: dict, expires_delta: timedelta | None = None):
#     to_encode = data.copy()
#     raw_expiry = settings.ACCESS_TOKEN_EXPIRES_IN
#     days_count = int(raw_expiry.replace("d", ""))
#     expire = datetime.now() + (expires_delta or timedelta(days=days_count))
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, settings.ACCESS_TOKEN_SECRET, algorithm="HS256")
#     return encoded_jwt


# def verify_access_token(token: str) -> dict:
#     """Verify JWT and return payload. Raises HTTPException if invalid."""
#     try:
#         payload = jwt.decode(
#             token,
#             settings.ACCESS_TOKEN_SECRET,
#             algorithms=["HS256"],
#         )
#         return payload
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired")
#     except jwt.InvalidTokenError:
#         raise HTTPException(status_code=401, detail="Invalid token")


# def hash_password(password: str) -> str:
#     """Hash plain password with Argon2. Use before storing in DB."""
#     return ph.hash(password)


# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """Verify plain password against hash. Use at login."""
#     try:
#         ph.verify(hashed_password, plain_password)
#         return True
#     except Exception:
#         return False
