import requests

url = 'http://localhost:5000/upload'
files = {'video': open('test.mp4', 'rb')}

response = requests.post(url, files=files)
print(response.text)