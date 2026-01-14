import requests

url = "http://localhost:8000/chat"
# Simulating a request from Company A
payload = {
    "message": "Give me a health report",
    "cpu": 92.5,
    "ram": 45.0,
    "company_id": "company_alpha_99"
}

try:
    response = requests.post(url, json=payload)
    print("--- API Response ---")
    print(response.json())
except Exception as e:
    print(f"Failed to connect: {e}")