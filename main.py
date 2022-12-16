from fastapi import FastAPI
from env import settings

from app.routes.api import api

app = FastAPI(title=settings.APP_NAME,
              description=settings.APP_DESCRIPTION,
              version=settings.APP_VERSION,
              docs_url=settings.LINK_DOCS,
              redoc_url=settings.LINK_REDOC)

app.include_router(api, prefix=settings.API_PREFIX)