# The Bandim Platform API

## Start Backend

```bash
cd backend;
sudo docker-compose build;
sudo docker-compose build up;
```

## Start Frontend

The example frontend page integrates a few of the endpoints of the backend for demonstration and testing purposes. To test the frontend page, change the hostname in the `example.html` file (to align with the locallay running backends server: `http://0.0.0.0:8000`) and then run these commands:

```bash
cd frontend;
python -m http.server --bind localhost 8080
```

Then point the browser to the address `localhost:8080` or `127.0.0.1:8080` to test a couple of the endpoints...
