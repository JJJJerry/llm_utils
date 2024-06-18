from flask import Flask, request
from flask_cors import cross_origin
from transformers import AutoTokenizer
import json
import torch
import datetime
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from argparse import ArgumentParser


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path, 
        trust_remote_code=True
    )
    if args.load_in_4bit:
        quant_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_quant_storage=None)
    else :
        quant_config=None
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    return model, tokenizer

app = Flask(__name__)

@app.route('/native', methods=['POST'])
@cross_origin()
def batch_chat():
    data = json.loads(request.get_data())
    query_and_histroy = {
        "query": data.get("prompt"),
        "history": data.get("history")
    }
    now = datetime.datetime.now()
    time_format = now.strftime("%Y-%m-%d %H:%M:%S")    
    try:
        query = query_and_histroy['query']
        history = query_and_histroy['history']
        output = model.chat(tokenizer=tokenizer,query=query,history=history)
        response = output[0]
        new_history = output[1] 
        new_history = history + [(query, response)]
        answer = {"response": response, "history": new_history, "status": 200, "time": time_format}    
        torch_gc()
        return answer
    except Exception as e:
        traceback.print_exc()
        torch_gc()
        return {"response": f"大模型预测出错:{repr(e)}", "history": [('', '')], "status": 444, "time": time_format}

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--load_in_4bit",action="store_true",default=False)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    with torch.no_grad():
        app.run(host='0.0.0.0', port=args.port)

