from pydantic import BaseSettings


class Settings(BaseSettings):
    # APPLICATION SETTING
    APP_NAME: str = "Patient Symptom NER"
    APP_DESCRIPTION: str = ("NER API Example on how to train pre-processed datasets into a Medical Machine Learning Model using Spacy as the ML Framework with FastAPI for API.")
    API_PREFIX: str = "/api/v1"
    APP_URL: str = "https://example.com"
    APP_VERSION: str = "v1.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_MODE: str = "development"
    LOG_LEVEL: str = "info"
    SECRET_KEY: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # DOCUMENTATION
    PUBLIC_URL: str = "public"
    STORAGE_URL: str = "storage"
    LINK_DOCS: str = "/docs"
    LINK_REDOC: str = "/redoc"

    # SECURITY
    ALLOWED_ORIGINS: str = "http://localhost:8080,http://127.0.0.1:8080"

    class Config:
        env_file = ".env"


settings = Settings()
