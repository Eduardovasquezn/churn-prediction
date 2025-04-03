import requests

data_to_send = {
    "customerid": "3164-YAXFY",
    "gender": "Male",
    "seniorcitizen": 0,
    "dependents": "No",
    "partner": "No",
    "contract": "Month-to-month",
    "tenure": 57,
    "paymentmethod": "Electronic check",
    "paperlessbilling": "Yes",
    "monthlycharges": 53.75,
    "totalcharges": 3196.0,
    "datetime_x": "2021-01-23 18:03:34.711729620",
    "deviceprotection": "Yes",
    "onlinebackup": "No",
    "onlinesecurity": "Yes",
    "internetservice": "DSL",
    "multiplelines": "No phone service",
    "phoneservice": "No",
    "techsupport": "No",
    "streamingmovies": "Yes",
    "streamingtv": "Yes",
    "datetime_y": "2021-01-23 18:03:34.711729620",
}

url = "http://127.0.0.1:8080/predict"
response = requests.post(url, json=data_to_send)

print(response.json())
