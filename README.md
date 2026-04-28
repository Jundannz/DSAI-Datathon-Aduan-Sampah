---
title: DSAI Datathon Aduan Sampah
emoji: 🗑️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Aduan Sampah AI — Text Classification & NER

API untuk klasifikasi prioritas aduan sampah dan ekstraksi lokasi (NER) dari teks laporan pengaduan.

## Endpoints

### Text Classification
**POST** `/predict/classification`
```json
{
  "text": "Sampah menumpuk di depan rumah sudah 3 hari tidak diangkut"
}
```
Response:
```json
{
  "input": "...",
  "prediction": "high_priority",
  "confidence": 0.92
}
```

### NER (Named Entity Recognition)
**POST** `/predict/ner`
```json
{
  "text": "Sampah di Jalan Kaliurang KM 8 sudah menumpuk"
}
```
Response:
```json
{
  "input": "...",
  "entities": [{"word": "Jalan Kaliurang KM 8", "label": "LOC"}]
}
```