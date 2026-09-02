import requests
import os

api_key = "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"
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())