from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from dtos.auth_dto import RegisterReqDTO, TokenResponseDTO
from helpers.auth_service import create_access_token, hash_password, verify_password
from core.dependencies import get_current_user
from models.users import User


router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)


@router.post("/register")
async def register(body: RegisterReqDTO):
    existing = await User.find_one(User.username == body.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed = hash_password(body.password)
    user = User(username=body.username, password=hashed)
    await user.insert()
    return {"message": "User registered successfully"}


@router.post("/login", response_model=TokenResponseDTO)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await User.find_one(User.username == form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(data={"sub": user.username})
    return TokenResponseDTO(access_token=token, token_type="bearer")


@router.get("/test")
async def test(payload=Depends(get_current_user)):
    return payload
