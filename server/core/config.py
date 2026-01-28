from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(".env")

if not ENV_FILE.exists():
    raise RuntimeError(".env file not found")


class Settings(BaseSettings):
    DATABASE_URL: str
    PORT: int
    OPENAI_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
