from io import StringIO

class InforProvider:
    def __init__(self, 
                 few_csv=4, # 少量和大量csv的分界点
                 ):
        self.few_csv = few_csv

    def df_info(self, df):
        output = StringIO()
        df.info(memory_usage=False, buf=output)
        info_str = output.getvalue()
        return info_str
    
    def df_columns(self, df):
        return str(df.columns.tolist())
    
    def df_head(self, df, n_lines=10):
        return str(df.head(n_lines))
    
    def df_markdown(self, df):
        return df.to_markdown(index=False)
    
    def df_content(
        self, 
        df, 
        print_1_1='数据全部内容信息：', 
        print_1_2='数据前几行内容信息：', 
        row_max=100, 
        column_max=20, 
        head_num=10
    ):

        rows, columns = df.shape

        if rows < row_max and columns < column_max:
            print_1 = print_1_1
            print_2 = df.to_csv(sep='\t', na_rep='nan')
        else:
            print_1 = print_1_2
            print_2 = df.head(head_num).to_csv(sep='\t', na_rep='nan')
            
        info = f"""{print_1}\n{print_2}"""
        return info

    def get_data_info_csv_few(self, csv_paths, csv_dfs):
        data_info_list = []
        if len(csv_paths) > 1:
            for i in range(len(csv_paths)):
                print_head = f'df{i+1}基本信息：'
                info_str = self.df_info(csv_dfs[csv_paths[i]])
                info_table = self.df_content(
                    df=csv_dfs[csv_paths[i]], 
                    print_1_1=f'df{i+1}全部内容信息：', 
                    print_1_2=f'df{i+1}前几行内容信息：'
                )
                data_info_list.append(f'{print_head}\n{info_str}\n{info_table}')
            data_info = "\n".join(data_info_list)
        else:
            info_str = self.df_info(csv_dfs[csv_paths[0]])
            info_table = self.df_content(csv_dfs[csv_paths[0]])
            data_info = f"数据基本信息：\n{info_str}\n{info_table}"
        
        return data_info
    
    def get_data_info_csv_many(self, csv_paths, csv_dfs):
        info = ""
        for file in csv_paths:
            info += f"""文件 {file} 的基本信息：\n"""
            info += self.df_info(csv_dfs[file]) + '\n'
            info += self.df_content(
                df=csv_dfs[file],
                print_1_1=f'文件 {file} 的全部内容信息：',
                print_1_2=f'文件 {file} 的前几行内容信息：', 
                row_max=50, 
                column_max=20, 
                head_num=5
            ) + '\n'
        return info
    
    def get_data_info_excel_few(self, excel_paths, excel_dfs):
        info = ""
        file = excel_paths[0]
        for sheet_name in excel_dfs[file]:
            info += f"""工作表 {sheet_name} 的基本信息：\n"""
            info += self.df_info(excel_dfs[file][sheet_name]) + '\n'
            info += self.df_content(
                df=excel_dfs[file][sheet_name],
                print_1_1=f'工作表 {sheet_name} 的全部内容信息：',
                print_1_2=f'工作表 {sheet_name} 的前几行内容信息：', 
                row_max=50, 
                column_max=20, 
                head_num=5
            ) + '\n'
        return info
    
    def get_data_info_excel_many(self, excel_paths, excel_dfs):
        info = ""
        for file in excel_paths:
            for sheet_name in excel_dfs[file]:
                info += f"""文件 {file} 中工作表 {sheet_name} 的基本信息：\n"""
                info += self.df_info(excel_dfs[file][sheet_name]) + '\n'
                info += self.df_content(
                    df=excel_dfs[file][sheet_name],
                    print_1_1=f'文件 {file} 中工作表 {sheet_name} 的全部内容信息：',
                    print_1_2=f'文件 {file} 中工作表 {sheet_name} 的前几行内容信息：', 
                    row_max=50, 
                    column_max=20, 
                    head_num=5
                ) + '\n'
        return info
    
    def get_data_info_many(self, csv_paths, excel_paths, csv_dfs, excel_dfs):
        info = ""
        for file in csv_paths:
            info += f"""文件 {file} 的基本信息：\n"""
            info += self.df_info(csv_dfs[file]) + '\n'
            info += self.df_content(
                df=csv_dfs[file],
                print_1_1=f'文件 {file} 的全部内容信息：',
                print_1_2=f'文件 {file} 的前几行内容信息：', 
                row_max=50, 
                column_max=20, 
                head_num=5
            ) + '\n'

        for file in excel_paths:
            for sheet_name in excel_dfs[file]:
                info += f"""文件 {file} 中工作表 {sheet_name} 的基本信息：\n"""
                info += self.df_info(excel_dfs[file][sheet_name]) + '\n'
                info += self.df_content(
                    df=excel_dfs[file][sheet_name],
                    print_1_1=f'文件 {file} 中工作表 {sheet_name} 的全部内容信息：',
                    print_1_2=f'文件 {file} 中工作表 {sheet_name} 的前几行内容信息：', 
                    row_max=50, 
                    column_max=20, 
                    head_num=5
                ) + '\n'
        return info
        
    
    def get_data_info(self, csv_paths, excel_paths, csv_dfs, excel_dfs):
        if len(csv_paths) <= self.few_csv and len(excel_paths) == 0:
            data_info = self.get_data_info_csv_few(csv_paths, csv_dfs)
        elif len(csv_paths) > self.few_csv and len(excel_paths) == 0:
            data_info = self.get_data_info_csv_many(csv_paths, csv_dfs)
        elif len(csv_paths) == 0 and len(excel_paths) == 1:
            data_info = self.get_data_info_excel_few(excel_paths, excel_dfs)
        elif len(csv_paths) == 0 and len(excel_paths) > 1:
            data_info = self.get_data_info_excel_many(excel_paths, excel_dfs)
        else:
            data_info = self.get_data_info_many(csv_paths, excel_paths, csv_dfs, excel_dfs)
        
        return data_info
