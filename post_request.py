import json
import requests


data = [
    {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
]

response = requests.post('https://scalable-ml-pipeline.onrender.com/predict', data=json.dumps(data))

print(response.status_code)
print(response.json()["predicted_salary"])