from contextlib import asynccontextmanager
from fastapi import FastAPI
from helpers.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient
from routes import base, data, nlp
from stores.llm.LLMProviderFactory import LLMProviderFactory
#  1. استدعاء الـ Factory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    llm_provider_factory = LLMProviderFactory(settings)
    #  2. إنشـاء كائن الـ Factory
    vectordb_provider_factory = VectorDBProviderFactory(settings)

    # Generation Client
    app.generation_client = llm_provider_factory.create(
        provider=settings.GENERATION_BACKEND
    )
    app.generation_client.set_generation_model(
        model_id=settings.GENERATION_MODEL_ID
    )

    # Embedding Client
    app.embedding_client = llm_provider_factory.create(
        provider=settings.EMBEDDING_BACKEND
    )
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    #  3. إنشاء الـ Vector DB Client وإسناده لـ app
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()

    
    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    yield

    # Shutdown Logic
    app.mongo_conn.close()
    if hasattr(app.vectordb_client, "disconnect"):
        app.vectordb_client.disconnect()


app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)