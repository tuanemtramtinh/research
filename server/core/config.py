from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(".env")

if not ENV_FILE.exists():
    raise RuntimeError(".env file not found")


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    # Optional for now because DB/auth are disabled.
    # When you re-enable those features, you can make them required again.
    MONGO_URL: Optional[str] = None
    DB_NAME: Optional[str] = None
    ACCESS_TOKEN_SECRET: Optional[str] = None
    ACCESS_TOKEN_EXPIRES_IN: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
