import httpx # pip install httpx
import json

# Test data to send to backend
test_data1 = [
  {
    "latitude": 1,
    "longitude": 0,
    "demand": 0,
    "depot": True
  },
  {
    "latitude": 2,
    "longitude": 0,
    "demand": 0,
    "depot": False
  },
  {
    "latitude": 3,
    "longitude": 0,
    "demand": 0,
    "depot": False
  },
  {
    "latitude": 4,
    "longitude": 0,
    "demand": 0,
    "depot": False
  },
  {
    "latitude": 5,
    "longitude": 0,
    "demand": 0,
    "depot": False
  }
]

if __name__ == "__main__":

    api_key =  "sp7b0JE3tg6RnhFRVsLVE1Vdsv2o7Riu12qlPHbcAOZPit30PdtIPWNi7gFDOahAB6GWilxWXik7qe5i4CLytIDaUwDOp4LnP00ZrdHS34YqKnXQAfp15TjQjWC3AHGP";

    # Create a new user
    httpx.request(
        method="POST",
        url="https://bandim.infonest.xyz/api/public/auth/register",
        headers={"X-API-Key": f"{api_key}"},
        data=json.dumps({"username": "user123", "password": "Passw0rd!"}),
    )

    # Login the user to retrieve an access key
    response = httpx.request(
        method="POST",
        url="https://bandim.infonest.xyz/api/public/auth/login",
        headers={"X-API-Key": f"{api_key}"},
        data=json.dumps({"username": "user123", "password": "Passw0rd!"}),
    )

    # Unpack the access token
    response_dict = response.json()
    access_token = response_dict["access_token"]

    # Insert the location data into the databse    
    response = httpx.request(
        method="POST",
        url="https://bandim.infonest.xyz/api/public/locations/bulk_insert",
        headers={"X-API-Key": f"{api_key}", "Authorization": f"Bearer {access_token}"},
        data=json.dumps(test_data1),
    )

    # Print the data we get back from the backend
    print(response.json())


