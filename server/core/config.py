from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(".env")

if not ENV_FILE.exists():
    raise RuntimeError(".env file not found")


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    MONGO_URL: str
    DB_NAME: str
    ACCESS_TOKEN_SECRET: str
    ACCESS_TOKEN_EXPIRES_IN: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
