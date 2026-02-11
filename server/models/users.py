from datetime import date
from typing import Optional
from beanie import Document
from pydantic import EmailStr, Field


class User(Document):
    # email: EmailStr
    # first_name: str
    # last_name: str
    # phone: Optional[str] = Field(default=None, max_length=11)
    # dob: date
    username: str
    password: str

    class Settings:
        name = "users"
