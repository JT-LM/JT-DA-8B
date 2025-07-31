from io import StringIO
import pandas as pd
import json
import re
import pandas as pd
import copy
from util.process_excel import process_excel_file 
import os
from da_workflow.preprocess import PreProcessor
from da_workflow.info_provide import InforProvider

class Sample:
    def __init__(self, 
                 id='',  
                 file_paths=None, 
                 vision_paths=None,
                 csv_paths=None,
                 excel_paths=None,
                 csv_dfs=None, 
                 excel_dfs=None, 
                 input_prompt='', 
                 history=None,  
                 generation='',
                 code='',
                 code_run_result='',
                 config=None,
                 add_info=None): 
        self.id = id
        self.file_paths = file_paths
        self.vision_paths = vision_paths
        self.csv_paths = []
        self.excel_paths = []
        self.csv_dfs = {}
        self.excel_dfs = {}
        self.input_prompt = input_prompt
        self.history = history
        self.generation = generation
        self.code = code
        self.code_run_result = code_run_result
        self.config = config
        self.add_info = {}

        self.few_csv = config.few_csv
        self.preprocessor = PreProcessor(config.cell_max_len)
        self.info_provider = InforProvider(config.few_csv)
        
        for path in self.file_paths:
            file_extension = os.path.splitext(path)[1].lower()
            if file_extension == ".csv":
                self.init_csv_dfs(path)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                self.init_excel_dfs(path)

        self.init_csv_excel_paths()
        self.df_preprocess()
        self.save_preprecess_file()
        # self.history_filter()
        
    def init_csv_dfs(self, path):
        df = pd.read_csv(path)
        self.csv_dfs[path] = copy.deepcopy(df)
        
    def init_excel_dfs(self, path):
        try:
            self.excel_dfs[path] = process_excel_file(path)
        except Exception as e:
            print(e)
            excel_file = pd.ExcelFile(path)
            sheet_names = excel_file.sheet_names
            self.excel_dfs[path] = {}
            for sheet_name in sheet_names:
                df = excel_file.parse(sheet_name)
                self.excel_dfs[path][sheet_name] = df
    
    def init_csv_excel_paths(self):
        self.csv_paths = list(self.csv_dfs.keys())
        self.excel_paths = list(self.excel_dfs.keys())
        
    def __str__(self):
        return str(vars(self))
    
    def __repr__(self):
        return str(vars(self))
    
    def df_preprocess(self):
        for path in self.csv_dfs:
            df = self.csv_dfs[path]
            df = self.preprocessor.strip_columns(df)
            df = self.preprocessor.fillna(df)
            df = self.preprocessor.cut_content(df)
            self.csv_dfs[path] = df
        
        for path in self.excel_dfs:
            for sheet_name in self.excel_dfs[path]:
                df = self.excel_dfs[path][sheet_name]
                df = self.preprocessor.fillna(df)
                df = self.preprocessor.cut_content(df)
                self.excel_dfs[path][sheet_name] = df
    
    def save_preprecess_file(self):
        def change_path(ori_path):
            file_name = os.path.splitext(path.split('/')[-1])[0]
            file_extension = os.path.splitext(path)[1].lower()
            new_file_path = f'/mnt/data/{file_name}{file_extension}'
            return new_file_path

        new_csv_dfs = {}
        for path in self.csv_dfs:
            df = self.csv_dfs[path]
            new_file_path = change_path(path)
            df.to_csv(new_file_path, index=False)
            new_csv_dfs[new_file_path] = copy.deepcopy(df)
        
        new_excel_dfs = {}
        for path in self.excel_dfs:
            all_df = self.excel_dfs[path]
            new_file_path = change_path(path)
            with pd.ExcelWriter(new_file_path) as writer:
                for sheet_name, df in all_df.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            new_excel_dfs[new_file_path] = copy.deepcopy(all_df)
        
        self.csv_dfs = new_csv_dfs
        self.excel_dfs = new_excel_dfs
        self.init_csv_excel_paths()
    
    def get_data_info(self):
        return self.info_provider.get_data_info(self.csv_paths, self.excel_paths, self.csv_dfs, self.excel_dfs)

    def append_history(self, history_msg):
        history_list = copy.deepcopy(history_msg)
        for i in range(len(history_list)):
            history_list[i]['data_path'] = self.file_paths
        self.history += history_list
        
    def get_history_msg(self):
        if self.history:
            history_msg = [{"role": history_item["role"], "content": history_item["content"]} for history_item in self.history]
        else:
            history_msg = []
        return history_msg
    
    def history_filter(self):
        if len(self.history) > 10:
            filtered_list = self.history[-10:]
            while len(filtered_list) > 0 and filtered_list[0].get("role") != "user":
                filtered_list.pop(0)
            self.history = filtered_list

        
            
