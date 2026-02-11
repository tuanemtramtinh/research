import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from models.diagram import Diagram
from models.users import User

# Nạp file .env
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

db_client = None


async def connect_db():
    global db_client

    try:
        db_client = AsyncIOMotorClient(MONGO_URL)
        await init_beanie(
            database=db_client[DB_NAME],
            document_models=[User, Diagram],
        )
        print("Connect to database successfully.")

    except Exception as e:
        logging.exception(f"Could not connect to mongo: {e}")
        raise


async def close_db_connect():
    global db_client

    if db_client:
        db_client.close()
        print("Close connection successfully.")


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await connect_db()
#     yield
#     await close_db_connect()
