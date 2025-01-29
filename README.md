# Setup
- You can user Visual studio configuration from `.devcontainer`  
- Run manually in docker container:
    - Build image: `docker build -t uwave_dev -f Dockerfile_dev .`
    - Run container: `docker run -it -p 5000:5000 --name uwave_dev_container uwave_dev`

# Run Server
`python ./src/server.py`

## Send requests
[httpie](https://httpie.io/) is used for these examples:
`http GET http://127.0.0.1:5000/model\?role\=user`  
`http GET http://127.0.0.1:5000/model\?id\=MODEL_ID\&role\=user`  
`http DELETE http://127.0.0.1:5000/model\?id\=1738138235871323\&role\=admin`  

`http POST http://127.0.0.1:5000/predict\?role\=user < data/inference_example.json`  
`http POST http://127.0.0.1:5000/predict\?id\=MODEL_ID\&role\=user < data/inference_example.json`  

`http POST http://127.0.0.1:5000/train\?role\=user`  

`http POST http://127.0.0.1:5000/dataset\?role\=user < data/dataset_update_example.json`  

# Run tests
`pytest ./tests/test_endpoints.py`  