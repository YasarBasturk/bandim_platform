import logging
from contextlib import asynccontextmanager
from sqlmodel import Session
from database import engine
from fastapi import FastAPI
from routers import public
from routers import default
from fastapi.middleware.cors import CORSMiddleware
from database import engine, redis_cache
from middleware import ContentSizeLimitMiddleware
from common import load_cors
from sqlmodel import SQLModel
from common import read_locations_from_json
from models import Location, DataSet
from sqlmodel import select


def create_db_and_tables():
    # Create database tables
    SQLModel.metadata.create_all(bind=engine)
    logging.info("Creating database and tables...")


async def insert_dataset(locations: dict[str, str]):
    dataset_name = "Random DataSet 1"
    with Session(engine) as session:
        logging.info("Creating an initial dataset...")
        statement = select(DataSet).where(DataSet.name == dataset_name)
        db_datasets = session.exec(statement).all()
        if len(db_datasets) == 0:
            dataset = DataSet.model_validate({"name": dataset_name})
            session.add(dataset)
            for location in locations:
                db_location = Location.model_validate(location)
                db_location.datasets = [dataset]
                session.add(db_location)
            session.commit()
        return None


async def start_up():
    # Create the database tables
    create_db_and_tables()
    # Populate the database with an initial dataset
    locations = read_locations_from_json("./random_geolocations.json")
    await insert_dataset(locations=locations)
    # Initialize redis cache
    await redis_cache.init_cache()


async def shut_down():
    await redis_cache.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Put any code here that needs to run at app START UP...
    await start_up()
    yield
    # Put any code here that needs to run at app SHUT DOWN...
    await shut_down()


tags_metadata = [
    {
        "name": "datasets",
        "description": "Operations on datasets, which primarily consist of one or more locations. Each dataset represents a logical grouping of locations.",
    },
    {
        "name": "locations",
        "description": "Operations on locations, which primarily consist of latitude and longitude coordinates. Each location represents the location of a household.",
    },
    {
        "name": "routes",
        "description": "Operations on rotues primarily consisting of an ordered sequence of locations. Each route is associated with a worker and a workplan.",
    },
    {
        "name": "workplans",
        "description": "Operations on workplans (associated with a specific dataset), which primarily consist of a collection of routes with additional time-specific information.",
    },
    {
        "name": "default",
        "description": "Retrieve API metadata",
    },
    {
        "name": "auth",
        "description": "Operations relating to the authentication of users.",
    },
    {
        "name": "users",
        "description": "Operations on users.",
    },
]

# app = FastAPI(docs_url = None, redoc_url = None)
# openapi_tags=openapi_tags
app = FastAPI(
    openapi_tags=tags_metadata,
    title="BandimPlatform",
    # description=description,
    summary="The Bandim Platform.",
    version="0.0.1",
    # terms_of_service="",
    # contact={
    #     "name": "",
    #     "url": "",
    #     "email": "",
    # },
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    lifespan=lifespan,
)

app.include_router(default.base, prefix="/api")

app.include_router(public.router, prefix="/api/public")


# Parameters & settings
logging.basicConfig(level=logging.DEBUG)

CORS_CONFIG = load_cors()
if CORS_CONFIG:
    app.add_middleware(CORSMiddleware, **CORS_CONFIG)

# Limit request size to 50_000 bytes ~ 0.05 megabytes
app.add_middleware(ContentSizeLimitMiddleware, max_content_size=5_00_00)


if __name__ == "__main__":
    create_db_and_tables()
    locations = read_locations_from_json("./random_geolocations.json")
