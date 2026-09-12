"""
ADVE Enterprise — Main FastAPI Server
Assembles middleware, routes, OpenAPI specification, and startup handlers.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.middleware.logging import StructlogMiddleware
from api.middleware.auth import verify_api_key
from api.middleware.license import verify_license_key
from api.routes import health, stream, batch, embeddings, config_route, compatibility

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    license_info = verify_license_key()
    print("=" * 60)
    print(f"  {settings.app_name}  v{settings.app_version}")
    print(f"  License Customer : {license_info.get('customer')}")
    print(f"  License Tier     : {license_info.get('tier', 'evaluation').upper()}")
    print(f"  Max Streams      : {license_info.get('max_streams')}")
    print(f"  Days Remaining   : {license_info.get('days_remaining')}")
    print("=" * 60)
    yield

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Commercial neural video encoding & vector embedding optimization engine.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/docs/openapi.json",
    lifespan=lifespan
)

# --- Middlewares ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StructlogMiddleware)

# --- Router Inclusions ---
app.include_router(health.router)
app.include_router(config_route.router, dependencies=[Depends(verify_api_key)])
app.include_router(stream.router, dependencies=[Depends(verify_api_key)])
app.include_router(batch.router, dependencies=[Depends(verify_api_key)])
app.include_router(embeddings.router, dependencies=[Depends(verify_api_key)])
app.include_router(compatibility.router, dependencies=[Depends(verify_api_key)])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
