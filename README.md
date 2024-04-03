# The Bandim Platform API

## Start Backend

```bash
cd backend;
sudo docker-compose build;
sudo docker-compose build up;
```

## Start Frontend

The example frontend page integrates a few of the endpoints of the backend for testing demonstration and testing purposes. Test the frontend page run:

```bash
cd frontend;
python -m http localhost 8080
```

Then point the browser to the address `localhost:8080` to test a couple of the endpoints. 