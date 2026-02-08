from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers.chat import router as chat_router
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):

    _ = settings
    print("Settings loaded & validated")

    yield

    print("App shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
