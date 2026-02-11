from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from controllers.auth import router as auth_router  # Auth temporarily disabled
from controllers.chat import router as chat_router
from core.config import settings

# from core.database import close_db_connect, connect_db  # Database temporarily disabled


@asynccontextmanager
async def lifespan(app: FastAPI):

    _ = settings
    print("Settings loaded & validated")

    # await connect_db()  # Database connection disabled for now

    yield

    # await close_db_connect()  # Database connection disabled for now

    print("App shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(auth_router)  # Auth routes disabled for now
app.include_router(chat_router)
