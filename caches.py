import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the API key (Replace with your actual API key)
GOOGLE_API_KEY = os.getenv("GENERATIVE_API_KEY")
if not GOOGLE_API_KEY:
    print("Please set the GENERATIVE_API_KEY in your environment.")

# Define the API endpoint
url = f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={GOOGLE_API_KEY}"

# Send a GET request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Print the response data (JSON format)
    print(response.json())
else:
    # Print the error if the request failed
    print(f"Request failed with status code {response.status_code}")
