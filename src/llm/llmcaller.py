import requests
import json
import os
import asyncio
import aiohttp
import transformers
from transformers import AutoTokenizer



class LLMCallerStreamMindIE(object):
    def __init__(self, config):
        from transformers import AutoTokenizer
        self.config = config
        self.use_model = 'JT-DA-8B'
        self.url = ""
        
        self.tokenizer_path = config.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = 'left'
        self.should_stop = False
        self.request_headers = {
                'content-type': 'application/json',
                }
    
    async def _async_call_stream(self, messages, is_first=False): 
        #print(messages)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=is_first)
        if prompt.endswith('<|im_end|>\n'):
            prompt = prompt[:-len('<|im_end|>\n')]
        # print('prompt:=====', prompt)
        # print('=======')
        data = {
            "inputs": prompt,
                "parameters": {
                    "do_sample": False,
                    "max_new_tokens": 16384,
                    "repetition_penalty": 1.1
                }
        }
        #with requests.post(self.url, stream=True, data=json.dumps(data)) as r:
        for i in range(3):
            try:
                with requests.post(self.url, stream=True, headers=self.request_headers, data=json.dumps(data)) as r:    
                    r.raise_for_status()
                    for event in r.iter_lines():
                        if event:
                            event = event.decode('utf-8')
                            if event[0:4] == "data" and event[6] == "{":
                                resJson = json.loads(event[5:])
                                # print(resJson)
                                if self.should_stop:
                                    r.close()
                                    break
                                yield {'choices': [{"text": resJson['token']['text']}]}
                return
            except Exception as e:
                print('call llm', i, str(e))
        
    