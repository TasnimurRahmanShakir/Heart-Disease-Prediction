import os
import sys

# Add the Backend folder to sys.path

from django.core.wsgi import get_wsgi_application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount, Router
from medScan.views import router

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Backend.settings')

django_app = get_wsgi_application()  # WSGI app here, not ASGI

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
fastapi_app.include_router(router, tags=["medScan"])

application = Router(
    routes=[
        Mount("/api", app=fastapi_app),
        Mount("/", app=WSGIMiddleware(django_app)),
    ]
)
