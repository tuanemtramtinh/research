from contextlib import asynccontextmanager
from fastapi import FastAPI
from controllers.chat import router as chat_router
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):

    _ = settings
    print("Settings loaded & validated")

    yield

    print("App shutting down")


app = FastAPI(lifespan=lifespan)

app.include_router(chat_router)
