from fastapi import APIRouter
from starlette.responses import Response
from settings import VERSION 

base = APIRouter()

@base.get("/version", tags=["default"])
async def version() -> Response:
    return Response(VERSION, 200)

