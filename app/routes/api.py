from fastapi import APIRouter
from app.controllers import (
    ner_controller
    )

api = APIRouter()

api.include_router(
    ner_controller.router,
    tags=["ner"])
