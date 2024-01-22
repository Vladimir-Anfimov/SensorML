# SensorML

## How to run this app

- Server
```console
cd server
pip install torch scikit-learn prophet matplotlib "fastapi[all]" pandas numpy
uvicorn api:app
```

- Client
```console
cd client
npm run dev
# press "o + <Enter>" to open in the brower
# alternatively you can type http://localhost:5173 in the browser
```
