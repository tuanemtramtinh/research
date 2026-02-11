# from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer

# from helpers.auth_service import verify_access_token
# from models.users import User


# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     object = verify_access_token(token)
#     username = object.get("sub")
#     current_user = await User.find_one(User.username == username)

#     if current_user is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
#         )

#     return current_user
