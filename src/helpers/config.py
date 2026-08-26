from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str
    APP_VERSION: str
    GROQ_API_KEY: Optional[str] = None

    # File Settings
    FILE_ALLOWED_TYPES: List[str]
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    # Database Settings
    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str

    # LLM & Models Settings
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    GENERATION_MODEL_ID: str
    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_SIZE: int

    # Defaults
    INPUT_DAFAULT_MAX_CHARACTERS: int
    GENERATION_DAFAULT_MAX_TOKENS: int
    GENERATION_DAFAULT_TEMPERATURE: float

    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None

    
    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"


    # Config (Pydantic v2)
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # لتجاهل أي متغيرات زوائد مستقبلاً بدون رفع Exception
    )


def get_settings() -> Settings:
    return Settings()