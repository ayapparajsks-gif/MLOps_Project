import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import os
import nest_asyncio


nest_asyncio.apply()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MLFLOW_EXPERIMENT_NAME="Retail_customer_segmentation"

#
DATA_PATH="/home/vboxuser/Downloads/customer_shopping_data.csv"
MODEL_DIR="models"
MODEL_PATH=os.path.join(MODEL_DIR,"segmentation_model.pkl")
SCALER_PATH=os.path.join(MODEL_DIR,"scaler.pkl")

try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logging.info(f"MLflow experiment is set to '{MLFLOW_EXPERIMENT_NAME}'")
except Exception as e:
    logging.error(f"Could not set MLflow experiment. Error: {e}")

def train_segmentation_model():
    with mlflow.start_run() as run:
        logging.info("Starting a new MLFlow run...")

        try:
            df=pd.read_csv(DATA_PATH)
            logging.info(f"Successfully loaded data from {DATA_PATH}")
        except FileNotFoundError:
            logging.error(f"Dataset not found at {DATA_PATH}")
            return None

        features = df[['age','price']]

        scaler=StandardScaler()
        features_scaled=scaler.fit_transform(features)

        num_clusters=4
        kmeans=KMeans(n_clusters=num_clusters,random_state=42,n_init=10)
        kmeans.fit(features_scaled)
        logging.info("Model training complete")

        mlflow.log_param("num_clusters",num_clusters)
        mlflow.log_param("features","age,price")
        mlflow.sklearn.log_model(kmeans,"kmeans_segmentation_model")
        mlflow.sklearn.log_model(scaler,"standard_scaler")
        
        logging.info("Model and scaler logged to MLflow successfully.")

        os.makedirs(MODEL_DIR,exist_ok=True)
        joblib.dump(kmeans,MODEL_PATH)
        joblib.dump(scaler,SCALER_PATH)
        logging.info("Model and scaler saved locally in '{MODEL_DIR}/' directory")
        
        return run.info.run_id


app=FastAPI(title="Retail Customer Insights API",version="1.0")

model=None
scaler=None

try:
    model=joblib.load(MODEL_PATH)
    scaler=joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded for API.")
except FileNotFoundError:
    logging.warning(f"Model/scaler not found. Train the model first.")
except Exception as e:
    logging.error(f"An error occured while loading model/scaler: {e}")

class CustomerInput(BaseModel):
    age:int
    price:float

@app.get("/",tags=["General"])
def root():
    return{"message":"Welcome to the Retail customer insights API. Use/docs to see endpoints."}

@app.post("/customer_segment",tags=["Prediction"])
def get_customer_segment(customer: CustomerInput):
    if not model or not scaler:
        raise HTTPException(status_code=503,detail="Model is not available. Please train it first.")

    try:
        input_data=pd.DataFrame([[customer.age,customer.price]],columns=['age','price'])
        input_scaled=scaler.transform(input_data)
        prediction=model.predict(input_scaled)
        segment_mapping={0:"Low-value",1:"Mid-value",2:"High-value",3:"VIP"}
        segment=segment_mapping.get(int(prediction[0]),"Unknown Segment")
        return{"customer_segment":segment,"segment_id":int(prediction[0])}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500,detail="An internal error occured during prediction.")
    

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("--- Models not found. Training customer segmentation model ---")
        train_segmentation_model()
    else:
        print("--- Models found, skipping training. ---")
    
    print("\n--- Starting FastAPI Server ---")
    uvicorn.run(app,host="0.0.0.0",port=8000)
