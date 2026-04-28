from transformers import pipeline

# Load pipeline untuk Text Classification
classification_pipeline = pipeline(
    "text-classification", 
    model="./models/text-classification", 
    device="cpu"
)

# Load pipeline untuk NER
ner_pipeline = pipeline(
    "token-classification", 
    model="./models/NER", 
    aggregation_strategy="simple", 
    device="cpu"
)

def get_predictions(text: str):
    # Eksekusi teks ke dalam model
    class_result = classification_pipeline(text)
    ner_result = ner_pipeline(text)
    
    # Perbaikan: Konversi tipe data numpy.float32 menjadi float standar Python
    # untuk hasil Text Classification
    for item in class_result:
        if "score" in item:
            item["score"] = float(item["score"])
            
    # Perbaikan: Konversi tipe data numpy.float32 dan numpy.int64 menjadi standar Python
    # untuk hasil NER (NER biasanya mengembalikan start dan end index karakter)
    for item in ner_result:
        if "score" in item:
            item["score"] = float(item["score"])
        if "start" in item:
            item["start"] = int(item["start"])
        if "end" in item:
            item["end"] = int(item["end"])
    
    return class_result, ner_result