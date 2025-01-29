import os
import shutil

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from metaflow import Flow, Run
from sklearn.preprocessing import StandardScaler


class UwaveService:
    def __init__(self):
        self.load_latest_model()
        self.dataset_path = "data/gesture_data.parquet"

    def load_latest_model(self):
        try:
            self.run = Flow("UwaveFlow").latest_successful_run
        except Exception as e:
            print("No trained model found: ", e)
            self.run = None

    def preprocess_input(self, input_data):
        """Preprocess the input data."""
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_data)
        return input_scaled

    def get_model(self, model_id):
        run = self.get_run(model_id)
        return run.data.model

    def get_run(self, model_id):
        try:
            run = Run(f"UwaveFlow/{model_id}")
        except Exception as e:
            print(e)
            raise ValueError(f"Model with ID {model_id} not found")
        return run.data

    def delete_model(self, model_id):
        try:
            # Assuming local storage for this example
            run_path = os.path.join(".metaflow", "UwaveFlow", str(model_id))
            if os.path.exists(run_path):
                shutil.rmtree(run_path)
            else:
                raise ValueError(f"Run data for model ID {model_id} not found")
        except Exception as e:
            print(e)
            raise ValueError(f"Model with ID {model_id} not found")

    def make_predictions(self, input_data, model_id):
        """Make predictions using the loaded model."""
        input_scaled = self.preprocess_input(input_data)

        model = self.get_model(model_id)

        predictions = model.predict(input_scaled)
        return predictions

    def train_model(self):
        """Run the UwaveFlow to train the model."""
        from subprocess import run

        run(["python", "src/uwave_flow.py", "run", "--dataset_path", self.dataset_path])
        self.load_latest_model()

    def update_dataset(self, input_data):
        # Data validation should be done here
        data = pd.read_parquet(self.dataset_path)
        data = pd.concat([data, input_data], ignore_index=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        suffix = ".parquet"
        new_dataset_path = self.dataset_path.split(suffix)[0] + "_" + timestamp + suffix
        data.to_parquet(new_dataset_path)
        self.dataset_path = new_dataset_path


print("Loading uwave model service")
service = UwaveService()
print("Model service loaded")

print("Starting FastAPI server")
app = FastAPI()


@app.post("/predict")
async def predict(request: Request, model_id: int = Query(None)):
    if service.run is None:
        raise HTTPException(status_code=404, detail="No trained model found")
    else:
        # Load JSON input
        input_json = await request.json()

        # Convert JSON data to DataFrame
        input_data = pd.DataFrame(input_json)

        if model_id is None:
            if service.run is None:
                raise HTTPException(status_code=404, detail="No trained models")
            model_id = service.run.id

        # Make predictions
        try:
            predictions = service.make_predictions(input_data, model_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Convert predictions to a list
        predictions_list = predictions.tolist()

        return {"predictions": predictions_list, "model_id": model_id}


@app.post("/train")
async def train():
    # Train the model
    # Hardening could be added in case model training fails
    service.train_model()
    return {"model_id": service.run.id}


@app.get("/model")
async def get_model(model_id: int = Query(None)):
    if model_id is None:
        if service.run is None:
            raise HTTPException(status_code=404, detail="No trained models")
        model_id = service.run.id
    try:
        run = service.get_run(model_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "model_id": model_id,
        "classification_report": run.classification_report,
        "confusion_matrix": run.confusion_matrix,
    }


@app.delete("/model")
async def delete_model(model_id: int = Query(None)):
    if model_id is None:
        raise HTTPException(status_code=400, detail="Model ID not provided")

    if str(model_id) == service.run.id:
        raise HTTPException(
            status_code=400, detail="Can not delete currently active model"
        )

    try:
        service.delete_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"status": "deleted"}


@app.post("/dataset")
async def update_dataset(request: Request):
    # Load JSON input
    input_json = await request.json()

    # Convert JSON data to DataFrame
    input_data = pd.DataFrame(input_json)

    try:
        service.update_dataset(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "updated"}


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=5000)
