## Overview

This API allows hospitals and healthcare providers to use existing patient symptom data and receive a predicted disease diagnosis.  
It integrates easily with Electronic Health Record (EHR) systems or healthcare platforms.

## Base URL

'http://127.0.0.1:5000'

## Authentication

All requests require an **API Key** sent in the request header:

x-api-key: CPSC597

If missing or invalid, the server responds with HTTP `401 Unauthorized`.

## Endpoints

### POST `/predict`

**Request Headers**
| Key          | Value                   |
|:-------------|:-------------------------|
| Content-Type | application/json          |
| x-api-key    | CPSC597                   | 

**Request Body Example**
Currently running with simulated EHR data. Please update data/ehr_data.csv with real EHR data.

```json
{
  "symptoms": ["fever", "cough", "fatigue"],
  "patient_id": "PT-1001",
  "include_top3": true
}

{
  "prediction": "Influenza",
  "confidence": 0.89,
  "top3": [
    {"disease": "Influenza", "confidence": 0.89},
    {"disease": "Common Cold", "confidence": 0.07},
    {"disease": "COVID-19", "confidence": 0.04}
  ],
  "symptoms_analyzed": 3
}
