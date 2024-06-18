import asyncio
import argparse
from utils import async_post_openai,load_json,save_json
class GetAnswer:
    def __init__(self,question_path,answer_path):
       self.question_path=question_path
       self.answer_path=answer_path
       self.res_json=[]
    async def run(self):
        self.questions_json=load_json(self.question_path)
        tasks=[asyncio.create_task(async_post_openai(question['instruction'])) for question in self.questions_json]
        answers= await asyncio.gather(*tasks) # answer是按顺序返回的，和prompt能对上
        assert len(answers)==len(self.questions_json)
        for i,answer in enumerate(answers):
            self.res_json.append( 
            {
                "instuction":self.questions_json[i]['instruction'],
                "input":"",
                "output":answer
            }) # 构建sft数据集
        save_json(self.res_json,self.answer_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='get sft dataset answer')
    parser.add_argument('--input_path', type=str,default='example_input_data.json')
    parser.add_argument('--output_path', type=str,default='example_output_data.json')
    args = parser.parse_args()
    getAnswer=GetAnswer(args.input_path,args.output_path)
    asyncio.run(getAnswer.run())