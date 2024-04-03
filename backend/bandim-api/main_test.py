from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from main import app
import pytest
from datetime import datetime, timedelta
from routers import public
from common import read_locations_from_json
from settings import API_KEYS


@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session

    app.dependency_overrides[public.get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# def test_security_jwt_auth(client: TestClient):
#     response = client.post("/api/public/auth/login")
#     assert response.status_code == 200, response.text


# def test_security_jwt_access_bearer(client: TestClient):
#     access_token = client.post("/api/public/auth/login").json()["access_token"]

#     response = client.get(
#         "/api/public/users/me", headers={"Authorization": f"Bearer {access_token}"}
#     )
#     assert response.status_code == 200, response.text
#     assert response.json() == {"username": "username", "role": "user"}


def test_security_jwt_access_bearer_wrong(client: TestClient):
    response = client.get(
        "/api/public/users/me",
        headers={
            "Authorization": "Bearer wrong_access_token",
            "X-API-Key": API_KEYS[0],
        },
    )
    assert response.status_code == 401, response.text


def test_security_jwt_access_bearer_no_credentials(client: TestClient):
    response = client.get("/api/public/users/me", headers={"X-API-Key": API_KEYS[0]})
    assert response.status_code == 401, response.text
    assert response.json() == {"detail": "Credentials are not provided"}


def test_security_jwt_access_bearer_incorrect_scheme_credentials(client: TestClient):
    response = client.get(
        "/api/public/users/me",
        headers={"Authorization": "Basic notreally", "X-API-Key": API_KEYS[0]},
    )
    assert response.status_code == 401, response.text
    assert response.json() == {"detail": "Credentials are not provided"}
    # assert response.json() == {"detail": "Invalid authentication credentials"}


# def test_security_jwt_refresh_bearer(client: TestClient):
#     refresh_token = client.post("/api/public/auth/login").json()["refresh_token"]

#     response = client.post(
#         "/api/public/auth/refresh", headers={"Authorization": f"Bearer {refresh_token}"}
#     )
#     assert response.status_code == 200, response.text


def test_security_jwt_refresh_bearer_wrong(client: TestClient):
    response = client.post(
        "/api/public/auth/refresh",
        headers={
            "Authorization": "Bearer wrong_refresh_token",
            "X-API-Key": API_KEYS[0],
        },
    )
    assert response.status_code == 401, response.text


def test_security_jwt_refresh_bearer_no_credentials(client: TestClient):
    response = client.post(
        "/api/public/auth/refresh", headers={"X-API-Key": API_KEYS[0]}
    )
    assert response.status_code == 401, response.text
    assert response.json() == {"detail": "Credentials are not provided"}


def test_security_jwt_refresh_bearer_incorrect_scheme_credentials(client: TestClient):
    response = client.post(
        "/api/public/auth/refresh",
        headers={"Authorization": "Basic notreally", "X-API-Key": API_KEYS[0]},
    )
    assert response.status_code == 401, response.text
    assert response.json() == {"detail": "Credentials are not provided"}
    # assert response.json() == {"detail": "Invalid authentication credentials"}


def create_user(client: TestClient):
    req = {
        "username": "Peter_Anderson",
        "password": "Passw0rd!",
    }
    response = client.post(
        "/api/public/auth/register", json=req, headers={"X-API-Key": API_KEYS[0]}
    )
    res = response.json()
    assert response.status_code == 200
    assert "uid" in res
    assert res["uid"] is not None
    assert res["is_active"] is True
    assert res["is_superuser"] is False
    return req


def login_user(client: TestClient):
    user_data = create_user(client=client)
    req = {
        "username": user_data["username"],
        "password": user_data["password"],
    }
    response = client.post(
        "/api/public/auth/login", json=req, headers={"X-API-Key": API_KEYS[0]}
    )
    res = response.json()
    assert "access_token" in res
    assert "refresh_token" in res
    access_token = res["access_token"]
    header = {"Authorization": f"Bearer {access_token}", "X-API-Key": API_KEYS[0]}
    return header


def create_bulk_locations_succeed(client: TestClient, auth_headers: dict[str, str]):
    req = [
        {
            "latitude": 11.85345134655994,
            "longitude": -15.598089853772322,
        },
        {
            "latitude": 11.84812109448719,
            "longitude": -15.600460066985532,
        },
    ]
    response = client.post(
        "/api/public/locations/bulk_insert",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    assert len(res) == 2
    assert "uid" in res[0]
    assert "uid" in res[1]
    return res


def test_bulk_locations_succeed(client: TestClient):
    auth_headers = login_user(client=client)
    create_bulk_locations_succeed(client=client, auth_headers=auth_headers)


def test_create_bulk_locations_fail(client: TestClient):
    auth_headers = login_user(client=client)
    # Create error: Missing longitude coordinates
    req = [
        {
            "longitude": -15.598089853772322,
        },
        {
            "longitude": -15.600460066985532,
        },
    ]
    response = client.post(
        "/api/public/locations/bulk_insert",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 422


def create_location_succeed(client: TestClient, auth_headers: dict[str, str]):
    req = {
        "latitude": 11.85345134655994,
        "longitude": -15.598089853772322,
    }
    response = client.post(
        "/api/public/locations/",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    assert "uid" in res
    return res


def get_location_succeed(
    client: TestClient, location_uid, auth_headers: dict[str, str]
):
    response = client.get(
        f"/api/public/locations/{location_uid}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    print(res)
    assert "uid" in res
    assert "datasets" in res
    assert "depot" in res
    assert "routes" in res
    return res


def test_location_succeed(client: TestClient):
    auth_headers = login_user(client=client)
    res = create_location_succeed(client=client, auth_headers=auth_headers)
    get_location_succeed(
        client=client, location_uid=res["uid"], auth_headers=auth_headers
    )


def create_dataset_succeed(
    client: TestClient, dataset_name, locations, auth_headers: dict[str, str]
):
    req = {"name": dataset_name, "locations": locations}
    response = client.post(
        "/api/public/datasets/",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    assert "uid" in res
    assert "name" in res
    assert "locations" in res
    assert "uid" in res["locations"][0]
    assert "uid" in res["locations"][1]
    assert "created_at" in res
    assert "updated_at" in res
    return res


def get_dataset_succeed(
    client: TestClient, dataset_uid, locations, auth_headers: dict[str, str]
):
    response = client.get(
        f"/api/public/datasets/{dataset_uid}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    assert "uid" in res
    assert "name" in res
    assert "locations" in res
    assert "uid" in res["locations"][0]
    assert "uid" in res["locations"][1]
    location_uids = [loc["uid"] for loc in res["locations"]]
    for location in locations:
        assert location["uid"] in location_uids
    assert "created_at" in res
    assert "updated_at" in res
    return res


def create_workplan_succeed(
    client: TestClient, dataset_uid: str, auth_headers: dict[str, str]
):
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=1)
    req = {
        "dataset_uid": dataset_uid,
        "start_time": str(start_time),
        "end_time": str(end_time),
        "workers": 3,
    }
    response = client.post(
        "/api/public/workplans/",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    assert "uid" in res
    assert "start_time" in res
    assert "end_time" in res
    assert "workers" in res
    assert "dataset_uid" in res
    assert "updated_at" in res
    assert "routes" in res
    return res


def get_workplan_succeed(
    client: TestClient, workplan_uid: str, auth_headers: dict[str, str]
):
    response = client.get(
        f"/api/public/workplans/{workplan_uid}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    res = response.json()
    assert "uid" in res
    assert "start_time" in res
    assert "end_time" in res
    assert "workers" in res
    assert "dataset_uid" in res
    assert "updated_at" in res
    assert "routes" in res
    return res


def test_all_succeed(client: TestClient):
    auth_headers = login_user(client=client)
    locations_res = create_bulk_locations_succeed(
        client=client, auth_headers=auth_headers
    )
    locations = [{"uid": str(loc["uid"])} for loc in locations_res]
    dataset_res = create_dataset_succeed(
        client,
        dataset_name="Random Dataset 0",
        locations=locations,
        auth_headers=auth_headers,
    )
    get_dataset_succeed(
        client=client,
        dataset_uid=dataset_res["uid"],
        locations=locations,
        auth_headers=auth_headers,
    )
    workplan_res = create_workplan_succeed(
        client=client,
        dataset_uid=dataset_res["uid"],
        auth_headers=auth_headers,
    )
    get_workplan_succeed(
        client=client,
        workplan_uid=workplan_res["uid"],
        auth_headers=auth_headers,
    )


def test_route_assignment(client: TestClient):
    auth_headers = login_user(client=client)
    # Load many locations
    file_path = "./random_geolocations.json"
    locations = read_locations_from_json(file_path)

    # Bulk insert locatinos
    response = client.post(
        "/api/public/locations/bulk_insert",
        json=locations,
        headers=auth_headers,
    )
    assert response.status_code == 200
    locations_res = response.json()
    locations = [{"uid": str(loc["uid"])} for loc in locations_res]

    # Associate the locations with a dataset
    req = {"name": "Random Dataset 1", "locations": locations}
    response = client.post(
        "/api/public/datasets/",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 200
    dataset_res = response.json()

    # Using the data set with added locations, create a workplan
    # for a number of workers
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=6)
    req = {
        "dataset_uid": dataset_res["uid"],
        "start_time": str(start_time),
        "end_time": str(end_time),
        "workers": 3,
    }
    response = client.post(
        "/api/public/workplans/",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 200
    workplan_res = response.json()

    # Assign routes and schedules for 3 workers
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=1)
    req = {
        "uid": workplan_res["uid"],
    }
    response = client.post(
        "/api/public/workplans/assign",
        json=req,
        headers=auth_headers,
    )
    assert response.status_code == 200
    assignment = response.json()
