from function_calls.pyecharts_func import Function_Calls
from mypybox import LocalPyBoxManager
import re
import os
import base64
from PIL import Image
from io import BytesIO
import logging
logger = logging.getLogger(__name__)

from util.str_op import has_code

def filter_text(text, target_str):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if target_str not in line]
    filtered_text = '\n'.join(filtered_lines)
    return filtered_text

def base642figure(base64_image, save_path):
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        image.save(save_path)
    except Exception as e:
        print(f"base642figure error: {e}")


class IPythonSandBox:
    def __init__(self, config, sample):
        self.config = config
        self.few_csv = config.few_csv
        self.cwd = config.cwd
        self.session_id = config.session_id
        self.filesaving_pattern: Pattern = re.compile(r'(?:\.savefig|\.to_csv)\(\s*[\'"]([^\'"]+)[\'"]\s*')
        self.error_trace_cleanup = False
        self.error_trace_cleanup_pattern: Pattern = re.compile(r"(Cell In\[\d+\], line \d+\n(?:.*\n)*?)(?=\n)")
        kwargs = {"cwd": str(self.cwd)} if self.cwd is not None else {}
        self.pybox_manager = LocalPyBoxManager()
        self.box = self.pybox_manager.start(kernel_id=self.session_id, **kwargs)
        self.init_dfs(sample)
        self.init_pyecharts()
    
    def init_pyecharts(self):
        function_call = Function_Calls()
        load_code = function_call.create_chart_call
        try:
            res = self.execute(load_code)
            print('pyechart', res)
        except Exception as e:
            print(f"Execution error: {str(e)}")    

    def init_dfs(self, sample):
        if len(sample.csv_paths) == 0 and len(sample.excel_paths) == 0:
            return
        if len(sample.csv_paths) <= self.few_csv and len(sample.excel_paths) == 0:
            self.init_csv_few(sample)
        elif len(sample.csv_paths) > self.few_csv and len(sample.excel_paths) == 0:
            self.init_csv_many(sample)
        elif len(sample.csv_paths) == 0 and len(sample.excel_paths) == 1:
            self.init_excel_few(sample)
        elif len(sample.csv_paths) == 0 and len(sample.excel_paths) > 1:
            self.init_excel_many(sample)
        else:
            self.init_many(sample)
            
    def init_csv_few(self, sample):
        if len(sample.csv_paths) > 1:
            self.init_multiple_csv_few(sample)
        else:
            self.init_single_csv(sample)
    
    
    def init_multiple_csv_few(self, sample):
        load_code = f"""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
"""
        
        read_code_list = [
            f"df{index+1} = pd.read_csv('{path}')"
            for index, path in enumerate(sample.csv_paths)
        ]
        init_code = "\n".join(read_code_list)
        
        init_plt = f"""import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['FZHei-B01']"""
        res1 = self.execute(load_code)
        res2 = self.execute(init_code)
        res3 = self.execute(init_plt)
        return res1 + res2 + res3
    
    def init_single_csv(self, sample):
        
        load_code = f"""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
"""
        
        init_code = f"""df = pd.read_csv('{sample.csv_paths[0]}')"""
        
        init_plt = f"""import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['FZHei-B01']"""
        
        res1 = self.execute(load_code)
        res2 = self.execute(init_code)
        res3 = self.execute(init_plt)
        return res1 + res2 + res3

    def init_csv_many(self, sample):
        load_code = f"""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
"""
        init_code = f"""csv_files = {sample.csv_paths}"""
        init_plt = f"""import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['FZHei-B01']"""
        res1 = self.execute(load_code)
        res2 = self.execute(init_code)
        res3 = self.execute(init_plt)
        return res1 + res2 + res3
    
    def init_excel_few(self, sample):
        load_code = f"""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
"""
        init_code = f"""excel_file = pd.ExcelFile({sample.excel_paths[0]})
sheet_names = excel_file.sheet_names"""
        init_plt = f"""import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['FZHei-B01']"""
        res1 = self.execute(load_code)
        res2 = self.execute(init_code)
        res3 = self.execute(init_plt)
        return res1 + res2 + res3
    
    def init_excel_many(self, sample):
        load_code = f"""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
"""
        init_code = f"""excel_files = {sample.excel_paths}"""
        init_plt = f"""import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['FZHei-B01']"""
        res1 = self.execute(load_code)
        res2 = self.execute(init_code)
        res3 = self.execute(init_plt)
        return res1 + res2 + res3
    
    def init_many(self, sample):
        load_code = f"""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
"""
        init_code = f"""csv_files = {sample.csv_paths}
excel_files = {sample.excel_paths}"""
        init_plt = f"""import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['FZHei-B01']"""
        res1 = self.execute(load_code)
        res2 = self.execute(init_code)
        res3 = self.execute(init_plt)
        return res1 + res2 + res3
    
    def process(self, sample):
        try:
            code_blocks = has_code(sample.generation)

            if not code_blocks:
                res = 'Does not contain code component.'
                sample.code_run_result = {"type": "text", "text": res}
                return sample

            last_code = code_blocks[-1]
            last_code = filter_text(last_code, 'exit(')
            last_code = filter_text(last_code, 'import os')
            last_code = filter_text(last_code, 'import sys')
            last_code = filter_text(last_code, 'import shutil')
            last_code = filter_text(last_code, 'import pathlib')
            last_code = filter_text(last_code, 'from os')
            last_code = filter_text(last_code, 'from sys')
            last_code = filter_text(last_code, 'from shutil')
            last_code = filter_text(last_code, 'from pathlib')
            last_code = filter_text(last_code, "plt.rcParams['font.sans-serif']")
            sample.code = last_code
            
            res = self.execute(last_code)
            for item in res:
                if item['type'] == 'image_url':
                    import uuid
                    def random_suffix() -> str:
                        return str(uuid.uuid4().hex)
                    random_idx = random_suffix()
                    path = f'{self.cwd}session-{self.session_id}-{random_idx}.png'
                    img_base64 = item['image_url']['url'].replace('data:image/png;base64,', '')
                    base642figure(img_base64, path)
                    item['image_url'] = f'!()[{path}]'
            
            sample.code_run_result = res
            return sample
        except Exception as e:
            res = f"Processing error: {str(e)}"
            sample.code_run_result = {"type": "text", "text": res}
            return sample
    
    def execute(
        self,
        query: str,
    ):
        """Executes the given query in an IPython kernel and returns the result as content and artifacts.

        Args:
            query (str): The code to execute in the IPython kernel.
            run_manager (CallbackManagerForToolRun | None): A manager for tracking tool execution.

        Returns:
            tuple: A tuple containing the content (a list of strings or dictionaries) and artifacts (a list of Artifact objects).
        """
    
        try:
            res = self.box.run(code=query)
        except TimeoutError:
            return [{"type": "text", "text": "Execution timed out. Please try again."}]

        content = []
        artifact = []
        for part in res.data:
            # We cannot mix str with dict for now, as `langgraph.prebuilt.ToolNode.msg_content_output` will dump it to str otherwise.
            # So we need to specify the text parts as dict.
            if (text_part := part.get("text/plain")) is not None:
                content.append({"type": "text", "text": text_part})

            if (img_part := part.get("image/png")) is not None:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_part}"},
                    }
                )

        if res.error is not None:
            cleaned_error = self._extract_error_trace(res.error)
            content.append({"type": "text", "text": cleaned_error})
        # print('content', content)
        return content
    
    def _extract_error_trace(self, e) -> str:
        """Extract and clean the error trace if enabled.

        Args:
            e (ErrorContent): The error content returned by the IPython kernel.

        Returns:
            str: The cleaned error trace.
        """
        if self.error_trace_cleanup and (match := re.search(self.error_trace_cleanup_pattern, str(e))) is not None:
            first_part = match.group(0)
            return f"{first_part}\n{e.ename}: {e.evalue}\n"
        return str(e)
    
    def __del__(self):
        pass
        