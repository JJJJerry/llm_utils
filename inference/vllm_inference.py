from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description='vllm inference')
parser.add_argument('--model_path', type=str)
parser.add_argument('--fp16',type=bool,default=True)
args = parser.parse_args()
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=256) # 可以修改
model_path=args.model_path
if args.fp16:
    dtype='float16'
else :
    dtype='auto'

llm = LLM(model=model_path,tokenizer=model_path,gpu_memory_utilization=0.9,dtype=dtype) # 根据具体需求的显存来修改
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
prompts=[
    '你好',
    'hi',
    '今天的天气很'
]
""" 
vllm_input=[]
for q in prompts:
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": q}]
    text=tokenizer.apply_chat_template(messages,
                                       tokenize=False,
                                       add_generation_prompt=True)
    vllm_input.append(text)
outputs = llm.generate(vllm_input, sampling_params) # chat
"""
outputs = llm.generate(prompts, sampling_params) # generate

# 获取输出
res=[]
for i,output in enumerate(outputs):
    generated_text = output.outputs[0].text
    print(generated_text)


    
    