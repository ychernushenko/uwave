# Analyze


# Run
python ./src/uwave_flow.py run --dataset_path "data/gesture_data.parquet"

# Start server
fastapi dev ./src/server.py --port 5000  
python ./src/server.py

## Send requests
http GET http://127.0.0.1:5000/model
http GET http://127.0.0.1:5000/model\?model_id\=MODEL_ID

http POST http://127.0.0.1:5000/predict < data/inference_example.json
http POST http://127.0.0.1:5000/predict\?model_id\=MODEL_ID < data/inference_example.json

http POST http://127.0.0.1:5000/train\?model_id\=MODEL_ID

http POST http://127.0.0.1:5000/dataset < data/dataset_update_example.json