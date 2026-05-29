import requests
import os

API_KEY = "EOxkaMFhw7lSZg6x2Fs9XPOFDda6xvi2jDPMOwRwUjg5M3ztZnHvoYqK"
query = "people"

os.makedirs("dataset/human", exist_ok=True)

for i in range(50):
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=1&page={i+1}"
    headers = {"Authorization": API_KEY}

    data = requests.get(url, headers=headers).json()
    img_url = data["photos"][0]["src"]["large"]

    img_data = requests.get(img_url).content

    with open(f"dataset/human/people{i}.jpg", "wb") as f:
        f.write(img_data)

print("Done")
