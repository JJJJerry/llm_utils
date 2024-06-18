import requests
import json


if __name__ == '__main__':
    api_url = 'http://127.0.0.1:8000/native'
    data = {
  "prompt":"我刚刚说了什么？",
  "history":[
    ['你好', '你好！很高兴能和你交流。有什么我可以帮助你的吗？']
  ]
}
    response=requests.post(url=api_url,data=json.dumps(data))
    print(response.json())