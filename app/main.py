from fastapi import FastAPI
from app.schemas import PredictRequest
from app.inference import get_predictions

app = FastAPI(title="API Klasifikasi dan NER Aduan Sampah")

@app.post("/predict")
def predict(request: PredictRequest):
    input_text = request.text
    
    # Ambil hasil dari file inference.py
    class_res, ner_res = get_predictions(input_text)
    
    # Format output JSON sesuai panduan
    return {
        "input": input_text,
        "classification_prediction": class_res,
        "ner_prediction": ner_res
    }