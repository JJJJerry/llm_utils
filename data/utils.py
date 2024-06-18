import json
import aiohttp
from constant import *
def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data=json.load(f)
    return data
def save_json(data,path):
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    with open(path, mode='w', encoding='utf-8') as json_file:
        json_file.write(json_str)
async def async_post_openai(prompt):
    for i in range(MAX_TRIES):
        headers = {"Authorization": f"Bearer {API_KEY}"}
        payload={
            "model":"deepseek-chat",
            "messages":[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OPENAI_API_BASE_URL}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=600.0,
                ) as result:
                    result = await result.json()
                    content = result['choices'][0]['message']['content']
                    #print(content)
                    return content
        except Exception as e:
            print(f'提问 {prompt}时出错 --{e}')