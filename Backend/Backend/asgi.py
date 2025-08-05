"""
ASGI config for Backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount, Router
from medScan.views import router as medscan_router


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Backend.settings')

django_app = get_asgi_application()

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
fastapi_app.include_router(medscan_router)

application = Router(
    routes=[
        Mount("/api", app=fastapi_app),        # FastAPI mounted at /api
        Mount("/", app=WSGIMiddleware(django_app)),  # Django serves all other routes
    ]
)