import prompts.common_prompt as common_prompt

import time
from sample_format.sample import Sample

from da_workflow.file_reading_workflow import get_response_1, get_response_2
import re
from util.str_op import extract_text_and_code, code_run_result_parse, stream_output
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ProtocolError
import asyncio
from util.str_op import has_code

class POT_Reasoner(object):
    def __init__(self, config, sample):
        self.config = config
        
        from llm.llmcaller import LLMCallerStreamMindIE
        self.llmcaller = LLMCallerStreamMindIE(config)

        from table.Infer_Sandbox_Jupyter import IPythonSandBox
        self.sandbox = IPythonSandBox(config, sample)
        
    def filt_msgs(self, msgs):
        if len(msgs) > 10:
            filtered_list = msgs[-10:]
            while len(filtered_list) > 0 and filtered_list[0].get("role") != "user":
                filtered_list.pop(0)
            return filtered_list
        return msgs

    def detect_chart_intent(self, user_input):
        chart_keywords = [
            '可视化', 
        ]
        pattern = '|'.join(chart_keywords)
        if re.search(pattern, user_input):
            return True
        return False

    def prompt(self, sample, stream=True):
        """Create a prompt for the LLM to reason about the input_prompt."""
        
        user_prompt = sample.input_prompt
        
        if self.detect_chart_intent(user_prompt):
            system_prompt = [{"role": "system", "content": common_prompt.STREAM_VIS_SYSTEM_PROMPT}]
        else:
            system_prompt = [{"role": "system", "content": common_prompt.STREAM_PYTHON_SYSTEM_PROMPT}]
        print("sys_prompt", system_prompt)
        table_sense_content = None
        print('sample.file_paths', sample.file_paths)
        if len(sample.file_paths) > 0 and (not sample.history or sample.file_paths != sample.history[-1]['data_path']):
            response_1 = get_response_1(sample)
            response_2 = sample.get_data_info()
            response_3 = get_response_2(sample)
            
            response_1 = response_1.replace(self.config.uid+'#', '')
            response_2 = response_2.replace(self.config.uid+'#', '')
            response_3 = response_3.replace(self.config.uid+'#', '')           
            paths = str(sample.csv_paths + sample.excel_paths).replace(self.config.uid+'#', '')

            prompt_msg = [
                {"role": "user", "content": paths},
                {"role": "assistant", "content": response_1},
                {"role": "user", "content": response_2},
                {"role": "assistant", "content": response_3},
                {"role": "user", "content": sample.input_prompt},
            ]
            sample.append_history(prompt_msg)
            
            print(response_1)
            print(f'\n```CodeExecuteResult\n{response_2}\n```\n')
            print(response_3)
            print()
            table_sense_content = response_1 + f'\n```CodeExecuteResult\n{response_2}\n```\n' + response_3 + '\n'
                
            msgs = system_prompt + prompt_msg
        else:
            history_msg = sample.get_history_msg()
            prompt_msg = [
                {"role": "user", "content": sample.input_prompt},
            ]
            sample.append_history(prompt_msg)
            msgs = system_prompt + history_msg + prompt_msg
        msgs = self.filt_msgs(msgs)
        return msgs, table_sense_content
    
    def parse(self, sample, response):
        print(response)
        sample.generation = response
        sample.append_history([{"role": "assistant", "content": response}])
        return sample
    
    async def reason_stream(self, sample, think=True):
        prompt_msg = []
        prompt_msg, table_sense_content = self.prompt(sample, stream=True)
        flag_first = True
        
        if table_sense_content is not None:
            yield {'delta': table_sense_content, 'role': 'assistant', 'status': 'init', 'response': {'type': 'text', 'text': table_sense_content, 'status': 'init'}}
            
        state = 'thinking'
        flag_has_code = False
        run_code_resources = 6
        agg_text = ''
        while run_code_resources > 0:
            agg_response = ''
            agg_text = ''
            self.llmcaller.should_stop = False
            async for response in self.llmcaller._async_call_stream(prompt_msg, is_first=flag_first):
                choices = response.get("choices", [])
                if not choices:
                    break
                choice = choices[0]
                content = choice.get("text", "")
                if content:
                    agg_response += content
                    content = content.replace('<think>', '')
                    print(content, end="") 
                    if '```python' in content:
                        state = 'coding'
                    elif state == 'coding' and '```' in content:
                        state = 'texting'
                    
                    if state == 'thinking':
                        if '</think>' in content:
                            yield {"response":{"type":"bot","result":[],"title":"","text":"","status":"init"},"role":"assistant","status":"init"}
                            state = 'texting'
                        else:
                            yield {"response":{"type":"thinking","result":[{"delta":content,"type":"quote"}],"status":"init"},"role":"assistant","status":"init"}
                    elif state == 'coding':
                        yield {"delta": content, "response": {"type": "code", "title": "", "status": "init", "result": [{"id": 0, "title": "", "content": content, "type": "PYTHON"}]}, "role": "assistant", "status": "init"}
                    elif state == 'texting':
                        agg_text += content
                        yield {'delta': content, 'role': 'assistant', 'status': 'init', 'response': {'type': 'text', 'text': agg_text, 'status': 'init'}}
                    
                    code_blocks = has_code(agg_response)
                    if code_blocks:
                        flag_has_code = True
                        agg_response = extract_text_and_code(agg_response)
                        code = code_blocks[-1]
                        sample.generation += agg_response
                        tmp = sample.generation
                        sample.generation = agg_response
                        sample = self.sandbox.process(sample)
                        sample.generation = tmp
                        code_run_result = sample.code_run_result
                        res_print = code_run_result_parse(code_run_result)
                        if '![](/largemodel/llmstudio/fs/' in res_print:
                            content = f"""\n{res_print}\n"""
                        else:
                            content = f"""\n```CodeExecuteResult\n{res_print}\n```\n"""
                        sample.generation += content
                        print(content)
                        state = 'tool_outputing'
                        yield {'delta': '', 'role': 'assistant', 'status': 'finish', 'response': {'type': 'text', 'text': agg_text, 'status': 'finish'}}
                        yield {"delta": content, "response": {"type": "text", "text": content, "status": "finish"}, "role": "tool", "status": "finish"}
                        state = 'texting'
                        self.llmcaller.should_stop = True
                        break
                    else:
                        flag_has_code = False
            run_code_resources -= 1
            if not flag_has_code and agg_response:
                sample.generation += agg_response
            
            if flag_first:
                prompt_msg.append({"role": "assistant", "content": sample.generation})
                flag_first = False
            else:
                prompt_msg[-1] = {"role": "assistant", "content": sample.generation}
            
            if not flag_has_code:
                break
        print()

        yield {'delta': '[EOS]', "role":"assistant","status":"finish", 'finished': 'Stop', 'Usage': {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}, "response": {"type":"text","text":agg_text,"status":"finish"}}
        print('agg_text:', agg_text)
        sample.append_history([{"role": "assistant", "content": sample.generation}])
