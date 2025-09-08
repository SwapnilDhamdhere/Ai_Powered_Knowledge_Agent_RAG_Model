from fastapi import APIRouter
from app.routes.upload_routes import router as upload_router
from app.routes.ask_routes import router as ask_router

# Main API Router
api_router = APIRouter()

# Register sub-routers
api_router.include_router(upload_router, prefix="/api/docs", tags=["Document Upload"])
api_router.include_router(ask_router, prefix="/api/ask", tags=["Ask AI"])
