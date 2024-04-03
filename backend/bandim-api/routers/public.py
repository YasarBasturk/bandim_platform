from fastapi import APIRouter, Depends, Query, HTTPException
from sqlmodel import Session, select
from sqlalchemy import insert
from fastapi import Security
from jwt import (
    JwtAccessBearerCookie,
    JwtAuthorizationCredentials,
    JwtRefreshBearer,
)

from passlib.context import CryptContext
from settings import (
    JWT_SECRET,
    JWT_EXPIRATION_TIME,
    API_KEYS,
)
import datetime as dt
import numpy as np
from vrp_solver.vrp_solver import (
    VRP,
    TwoOptSolver,
    KMeansRadomizedPopulationInitializer,
    FitnessFunctionMinimizeDistance,
    individual_to_routes,
)
from database import engine
from models import (
    DataSet,
    DataSetCreate,
    DataSetReadCompact,
    DataSetReadDetails,
    DataSetUpdate,
    Location,
    LocationCreate,
    LocationReadCompact,
    LocationReadDetails,
    WorkPlan,
    WorkPlanReadCompact,
    WorkPlanReadDetails,
    WorkPlanCreate,
    Identifier,
    Route,
    RouteRead,
    RouteRead,
    RouteCreate,
    Timestamp,
    TimestampCreate,
    LocationTimestampReadDetails,
    LocationTimestampCollection,
    User,
    UserCreate,
    UserRead,
    TokenRead,
    UserLogin,
)
import uuid
import pandas as pd
from fastapi.security import APIKeyHeader
from common import check_api_key


router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Hasher:
    @staticmethod
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)


api_key_header = APIKeyHeader(name="X-API-Key")

# Read access token from bearer header and cookie (bearer priority)
access_security = JwtAccessBearerCookie(
    secret_key=JWT_SECRET,  # TODO: Add secret from env var
    auto_error=False,
    access_expires_delta=dt.timedelta(hours=JWT_EXPIRATION_TIME),
)

# Read refresh token from bearer header only
refresh_security = JwtRefreshBearer(
    secret_key=JWT_SECRET,
    auto_error=True,  # automatically raise HTTPException: HTTP_401_UNAUTHORIZED
)


def get_session():
    with Session(engine) as session:
        yield session


@router.post("/locations/", response_model=LocationReadCompact, tags=["locations"])
async def create_location(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    location: LocationCreate,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)
    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_location = Location.model_validate(location)
    session.add(db_location)
    session.commit()
    session.refresh(db_location)
    return db_location


@router.post(
    "/locations/bulk_insert",
    response_model=list[LocationReadCompact],
    tags=["locations"],
)
async def create_locations(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    locations: list[LocationCreate],
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_locations = []
    for location in locations:
        db_location = Location.model_validate(location)
        db_locations.append(db_location)
    statement = insert(Location).returning(Location)
    results = session.scalars(statement, db_locations)
    session.flush()
    return results.all()


@router.get("/locations/", response_model=list[LocationReadCompact], tags=["locations"])
async def read_locations(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=100, le=100),
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_locations = session.exec(select(Location).offset(offset).limit(limit)).all()
    return db_locations


@router.get(
    "/locations/{location_uid}", response_model=LocationReadDetails, tags=["locations"]
)
async def read_location(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    location_uid: uuid.UUID,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_location = session.get(Location, location_uid)
    if not db_location:
        raise HTTPException(status_code=404, detail="Location not found")
    return db_location


@router.delete("/datasets/{dataset_uid}", tags=["datasets"])
def delete_dataset(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    dataset_uid: uuid.UUID,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_dataset = session.get(DataSet, dataset_uid)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    session.delete(db_dataset)
    session.commit()
    return {"ok": True}


@router.post("/datasets/", response_model=DataSetReadDetails, tags=["datasets"])
async def create_dataset(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    dataset: DataSetCreate,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    primary_keys_list = [str(loc.uid) for loc in dataset.locations]
    statement = select(Location).where(Location.uid.in_(primary_keys_list))
    locations = session.exec(statement).all()
    dataset.locations = locations
    db_dataset = DataSet.model_validate(dataset)
    session.add(db_dataset)
    session.commit()
    session.refresh(db_dataset)
    return db_dataset


@router.get("/datasets/", response_model=list[DataSetReadCompact], tags=["datasets"])
async def read_datasets(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=100, le=100),
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_datasets = session.exec(select(DataSet).offset(offset).limit(limit)).all()
    return db_datasets


@router.get(
    "/datasets/{dataset_uid}", response_model=DataSetReadDetails, tags=["datasets"]
)
async def read_dataset(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    dataset_uid: uuid.UUID,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_dataset = session.get(DataSet, dataset_uid)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return db_dataset


# @router.patch(
#     "/datasets/{dataset_uid}", response_model=DataSetReadDetails, tags=["datasets"]
# )
# async def update_dataset(
#     *,
#     session: Session = Depends(get_session),
#     dataset_uid: uuid.UUID,
#     dataset: DataSetUpdate
# ):
#     db_dataset = session.get(DataSet, dataset_uid)
#     if not db_dataset:
#         raise HTTPException(status_code=404, detail="Dataset not found")
#     dataset_data = dataset.model_dump(exclude_unset=True)
#     for key, value in dataset_data.items():
#         setattr(db_dataset, key, value)
#     session.add(db_dataset)
#     session.commit()
#     session.refresh(db_dataset)
#     return db_dataset


@router.post("/workplans/", response_model=WorkPlanReadCompact, tags=["workplans"])
async def create_workplan(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    workplan: WorkPlanCreate,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_workplan = WorkPlan.model_validate(workplan)
    session.add(db_workplan)
    session.commit()
    session.refresh(db_workplan)
    # For consistency add (empty) route data associated with the workplan
    statement = select(Route).where(Route.workplan_uid == db_workplan.uid)
    db_routes = session.exec(statement).all()
    workplan_dict = db_workplan.model_dump()
    workplan_dict["routes"] = db_routes
    return WorkPlanReadCompact(**workplan_dict)


@router.get(
    "/workplans/{workplan_uid}", response_model=WorkPlanReadCompact, tags=["workplans"]
)
async def read_workplan(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    workplan_uid: uuid.UUID,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_workplan = session.get(WorkPlan, workplan_uid)
    if not db_workplan:
        raise HTTPException(status_code=404, detail="WorkPlan not found")
    # For consistency add (empty) route data associated with the workplan
    statement = select(Route).where(Route.workplan_uid == db_workplan.uid)
    db_routes = session.exec(statement).all()
    workplan_dict = db_workplan.model_dump()
    workplan_dict["routes"] = db_routes
    return WorkPlanReadCompact(**workplan_dict)


@router.post(
    "/workplans/assign", response_model=LocationTimestampCollection, tags=["workplans"]
)
async def assign_workplan(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    workplan: Identifier,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_workplan = get_workplan(session, workplan.uid)
    db_dataset = get_dataset(session, db_workplan.dataset_uid)
    locations, df = extract_locations(db_dataset)
    routes = solve_routing_problem(db_workplan, locations)
    route_assignments = create_and_store_routes(session, db_workplan, routes, df)
    return LocationTimestampCollection(assignments=route_assignments)


def get_workplan(session: Session, uid: str) -> WorkPlan:
    db_workplan = session.get(WorkPlan, uid)
    if not db_workplan:
        raise HTTPException(status_code=404, detail="WorkPlan not found")
    return db_workplan


def get_dataset(session: Session, dataset_uid: str) -> DataSet:
    db_dataset = session.get(DataSet, dataset_uid)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="DataSet not found")
    return db_dataset


def extract_locations(db_dataset: DataSet) -> tuple[list[dict], pd.DataFrame]:
    data = [
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "depot": loc.depot,
            "demand": loc.demand,
            "uid": loc.uid,
        }
        for loc in db_dataset.locations
    ]
    # Put the location data into a dataframe for easier sorting and filtering
    df = (
        pd.DataFrame(data=data)
        .astype(
            {
                "latitude": "float64",
                "longitude": "float64",
                "depot": "bool",
                "demand": "int64",
                "uid": "object",
            }
        )
        .sort_values(by="depot", ascending=False)
    )
    locations = df[["latitude", "longitude"]].to_numpy().tolist()
    return locations, df


def solve_routing_problem(db_workplan: WorkPlan, locations: list[dict]) -> list[dict]:
    # Create a vehicle routing problem (vrp) instance to be solved
    # The problem consists of a collection of locations that have to
    # be divided up and visited by a number of workers, while minimizing
    # the total distance
    vrp_instance = VRP(
        locations=locations,
        num_salesmen=db_workplan.workers,
        precompute_distances=True,
    )

    # Algorithm Parameter: Determine the appropriate population size
    n = len(locations)
    population_maximum = 10_000
    population_minimum = 25
    population_size = np.minimum(
        np.maximum(population_minimum, int(n / np.log2(n))), population_maximum
    )
    # Run the 2-opt solver on each individual of the initial population
    # to then find the best individual among the population
    solver = TwoOptSolver(
        vrp_instance=vrp_instance,
        population_size=population_size,
        population_initializer_class=KMeansRadomizedPopulationInitializer,
        fitness_function_class=FitnessFunctionMinimizeDistance,
    )

    # Run the 2-opt solver
    result = solver.run()
    # Retrieve the best solution in terms of distance
    # The best solution is a route (specific sequence of locations)
    # for each worker
    best_solution = result.get_topk(k=1)[0]
    routes = individual_to_routes(best_solution, vrp_instance)
    return routes


def create_and_store_routes(
    session: Session, db_workplan: WorkPlan, routes: list[dict], df: pd.DataFrame
) -> list[dict]:
    # Flatten route data into a DataFrame directly
    flattened_data = [
        (latitude, longitude, route_index, visit_number)
        for route_index, route in enumerate(routes)
        for visit_number, (latitude, longitude) in enumerate(route)
    ]
    ndf = pd.DataFrame(
        flattened_data, columns=["latitude", "longitude", "route", "visit_number"]
    ).astype(
        {
            "latitude": "float64",
            "longitude": "float64",
            "route": "int64",
            "visit_number": "int64",
        }
    )

    merged_dfs = pd.merge(df, ndf, on=["latitude", "longitude"]).sort_values(
        by=["route", "visit_number"],
        ascending=True,
    )

    algorithmrun_uid = uuid.uuid4()
    route_list = []
    for _, group_df in merged_dfs.groupby("route"):
        # Query locations in bulk (associated with a specific route)
        primary_keys_list = group_df["uid"].to_numpy().tolist()
        statement = select(Location).where(Location.uid.in_(primary_keys_list))
        locations = session.exec(statement).all()

        # Create a  `Route` object based on a list of `Location` objects
        db_route = Route(
            locations=locations,
            workplan_uid=db_workplan.uid,
            algorithmrun_uid=algorithmrun_uid,
        )
        session.add(db_route)

        # Generate and add Timestamps in bulk and which are to be associated with a
        # specific route and location
        current_time = db_workplan.start_time
        timestamps = [
            Timestamp(
                datetime=current_time + dt.timedelta(seconds=120 + 10 * row["demand"]),
                route_uid=db_route.uid,
                location_uid=row["uid"],
            )
            for _, row in group_df.iterrows()
        ]

        # Commit per route to encapsulate transactions
        session.bulk_save_objects(timestamps)
        session.commit()

        # Fetch all the timestamps for the locations in this route
        query = (
            select(Location, Timestamp.datetime)
            .join(Timestamp, Timestamp.location_uid == Location.uid)
            .where(Timestamp.route_uid == db_route.uid)
            .order_by(Timestamp.datetime)
        )
        results = session.exec(query).all()

        locations_with_timestamps = [
            {"location": result[0], "timestamp": result[1]} for result in results
        ]
        route_list.append(locations_with_timestamps)
    return route_list


# @router.post("/routes/", response_model=RouteRead, tags=["routes"])
# async def create_route(
#     *, session: Session = Depends(get_session), route: RouteCreate
# ):
#     db_route = WorkPlan.model_validate(route)
#     session.add(db_route)
#     session.commit()
#     session.refresh(db_route)
#     # For consistency add (empty) route data associated with the workplan
#     # statement = select(Route).where(Route.workplan_uid == db_workplan.uid)
#     # db_routes = session.exec(statement).all()
#     # workplan_dict = db_workplan.model_dump()
#     # workplan_dict["routes"] = db_routes
#     # return WorkPlanRead(**workplan_dict)
#     return db_route


@router.get("/routes/{route_uid}", response_model=RouteRead, tags=["routes"])
async def read_route(
    *,
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
    session: Session = Depends(get_session),
    route_uid: uuid.UUID,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")

    db_route = session.get(Route, route_uid)
    if not db_route:
        raise HTTPException(status_code=404, detail="Route not found")
    return db_route


@router.post(
    "/auth/register",
    response_model=UserRead,
    tags=["auth"],
)
async def register_user(
    *,
    x_api_key: str = Security(api_key_header),
    session: Session = Depends(get_session),
    user: UserCreate,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    statement = select(User).where(User.username == user.username)
    db_users = session.exec(statement).all()
    if len(db_users) > 0:
        raise HTTPException(status_code=400, detail="User already exists")
    db_user = User(
        **user.model_dump(), hashed_password=Hasher().get_password_hash(user.password)
    )
    db_user = User.model_validate(db_user)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return UserRead(**db_user.model_dump())


@router.post("/auth/login", response_model=TokenRead, tags=["auth"])
def auth(
    *,
    x_api_key: str = Security(api_key_header),
    session: Session = Depends(get_session),
    user: UserLogin,
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    statement = select(User).where(User.username == user.username)
    db_user = session.exec(statement).first()

    if db_user is None:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    verified = Hasher().verify_password(
        plain_password=user.password, hashed_password=db_user.hashed_password
    )

    if not verified:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # subject (actual payload) is any json-able python dict
    subject = {"username": db_user.username, "uid": str(db_user.uid)}

    # Create new access/refresh tokens pair
    access_token = access_security.create_access_token(subject=subject)
    refresh_token = refresh_security.create_refresh_token(subject=subject)

    return TokenRead(access_token=access_token, refresh_token=refresh_token)


@router.post("/auth/refresh", response_model=TokenRead, tags=["auth"])
def refresh(
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(refresh_security),
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # Update access/refresh tokens pair
    # We can customize expires_delta when creating
    access_token = access_security.create_access_token(subject=credentials.subject)
    refresh_token = refresh_security.create_refresh_token(
        subject=credentials.subject, expires_delta=dt.timedelta(days=2)
    )

    # return {"access_token": access_token, "refresh_token": refresh_token}
    return TokenRead(access_token=access_token, refresh_token=refresh_token)


@router.get("/users/me", tags=["users"])
def read_current_user(
    x_api_key: str = Security(api_key_header),
    credentials: JwtAuthorizationCredentials = Security(access_security),
):
    check_api_key(x_api_key=x_api_key, api_keys=API_KEYS)

    # auto_error=False. We should check manually
    if not credentials:
        raise HTTPException(status_code=401, detail="Credentials are not provided")
    
    # now we can access Credentials object
    return {"username": credentials["username"], "uid": credentials["uid"]}
