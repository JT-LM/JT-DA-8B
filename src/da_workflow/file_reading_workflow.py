import pandas as pd
import os
from sample_format.sample import Sample


def get_response_1_csv_few(sample):
    if len(sample.csv_paths) > 1:
        read_code_list = [
            f"df{index+1} = pd.read_csv('{path}')"
            for index, path in enumerate(sample.csv_paths)
        ]
        read_code_str = "\n".join(read_code_list)
        
        info_code_list = [
            f"""# 查看数据df{index+1}基本信息
print('df{index+1}基本信息：')
df{index+1}.info()

# 查看数据集行数和列数
rows{index+1}, columns{index+1} = df{index+1}.shape

if rows{index+1} < 100 and columns{index+1} < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('df{index+1}全部内容信息：')
    print(df{index+1}.to_csv(sep='\\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('df{index+1}前几行内容信息：')
    print(df{index+1}.head().to_csv(sep='\\t', na_rep='nan'))
""" 
            for index, path in enumerate(sample.csv_paths)
        ]
        info_code_str = "\n".join(info_code_list)
    else:
        read_code_str = f"df = pd.read_csv('{sample.csv_paths[0]}')"
        info_code_str = """print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\\t', na_rep='nan'))
"""
        
    response_text = """我已经收到您的数据文件，我需要查看数据文件内容以对数据集有一个初步的了解，并进行一些数据预处理，为进一步分析做准备。
首先我会读取数据，并了解数据基本情况。
然后我会根据数据量的大小查看数据的具体内容。
"""
    response_code = """```python
import pandas as pd

# 加载数据集
"""
    response_code += read_code_str
    response_code += '\n\n'
    response_code += f"""#查看数据基本信息
{info_code_str}
```"""
    response1 = response_text + response_code
    return response1

def get_response_1_csv_many(sample):
    response_text = """我已经收到您的数据文件，我需要查看数据文件内容以对数据集有一个初步的了解，并进行一些数据预处理，为进一步分析做准备。
首先我会读取数据，并了解数据基本情况。
然后我会根据数据量的大小查看数据的具体内容。
"""
    response_code = """```python
import pandas as pd

# 查看数据集
"""

    code_str = f"""csv_files = {sample.csv_paths}
for file in csv_files:
    df = pd.read_csv(file)
    
    print(f'文件 {{file}} 的基本信息：')
    df.info()

    # 查看数据集行数和列数
    rows, columns = df.shape

    if rows < 50 and columns < 20:
        # 短表数据（行数少于50且列数少于20）查看全量数据信息
        print(f'文件 {{file}} 的全部内容信息：')
        print(df.to_csv(sep='\t', na_rep='nan'))
    else:
        # 长表数据查看数据前几行信息
        print(f'文件 {{file}} 的前几行内容信息：')
        print(df.head().to_csv(sep='\t', na_rep='nan'))
```"""
    response_code += code_str
    response1 = response_text + response_code
    return response1
    
def get_response_1_excel_few(sample):
    response_text = """我已经收到您的数据文件，我需要查看数据文件内容以对数据集有一个初步的了解，并进行一些数据预处理，为进一步分析做准备。
首先我会读取数据，并了解数据基本情况。
然后我会根据数据量的大小查看数据的具体内容。
"""
    response_code = """```python
import pandas as pd

# 查看数据集
"""

    code_str = f"""excel_file = pd.ExcelFile({sample.excel_paths[0]})
sheet_names = excel_file.sheet_names
for sheet_name in sheet_names:
    df = excel_file.parse(sheet_name)

    print(f'工作表 {{sheet_name}} 的基本信息：')
    df.info()

    # 查看数据集行数和列数
    rows, columns = df.shape

    if rows < 50 and columns < 20:
        # 短表数据（行数少于50且列数少于20）查看全量数据信息
        print(f'工作表 {{sheet_name}} 的全部内容信息：')
        print(df.to_csv(sep='\t', na_rep='nan'))
    else:
        # 长表数据查看数据前几行信息
        print(f'工作表 {{sheet_name}} 的前几行内容信息：')
        print(df.head().to_csv(sep='\t', na_rep='nan'))
```"""
    response_code += code_str
    response1 = response_text + response_code
    return response1

def get_response_1_excel_many(sample):
    response_text = """我已经收到您的数据文件，我需要查看数据文件内容以对数据集有一个初步的了解，并进行一些数据预处理，为进一步分析做准备。
首先我会读取数据，并了解数据基本情况。
然后我会根据数据量的大小查看数据的具体内容。
"""
    response_code = """```python
import pandas as pd

# 查看数据集
"""

    code_str = f"""excel_files = {sample.excel_paths}
for file in excel_files:
    excel_file = pd.ExcelFile(file)
    sheet_names = excel_file.sheet_names
    sheet_names
    for sheet_name in sheet_names:
        df = excel_file.parse(sheet_name)

        print(f'文件 {{file}} 中工作表 {{sheet_name}} 的基本信息：')
        df.info()

        # 查看数据集行数和列数
        rows, columns = df.shape

        if rows < 50 and columns < 20:
            # 短表数据（行数少于50且列数少于20）查看全量数据信息
            print(f'文件 {{file}} 中工作表 {{sheet_name}} 的全部内容信息：')
            print(df.to_csv(sep='\t', na_rep='nan'))
        else:
            # 长表数据查看数据前几行信息
            print(f'文件 {{file}} 中工作表 {{sheet_name}} 的前几行内容信息：')
            print(df.head().to_csv(sep='\t', na_rep='nan'))
```"""
    response_code += code_str
    response1 = response_text + response_code
    return response1
    
def get_response_1_many(sample):
    response_text = """我已经收到您的数据文件，数据中包含csv和excel文件，我需要查看数据文件内容以对数据集有一个初步的了解，并进行一些数据预处理，为进一步分析做准备。
首先我会读取数据，并了解数据基本情况。
然后我会根据数据量的大小查看数据的具体内容。
"""
    response_code = """```python
import pandas as pd

"""

    code_str = f"""# 查看 CSV 文件
csv_files = {sample.csv_paths}
for file in csv_files:
    df = pd.read_csv(file)
    
    print(f'文件 {{file}} 的基本信息：')
    df.info()

    # 查看数据集行数和列数
    rows, columns = df.shape

    if rows < 50 and columns < 20:
        # 短表数据（行数少于50且列数少于20）查看全量数据信息
        print(f'文件 {{file}} 的全部内容信息：')
        print(df.to_csv(sep='\t', na_rep='nan'))
    else:
        # 长表数据查看数据前几行信息
        print(f'文件 {{file}} 的前几行内容信息：')
        print(df.head().to_csv(sep='\t', na_rep='nan'))

# 查看 Excel 文件
excel_files = {sample.excel_paths}
for file in excel_files:
    excel_file = pd.ExcelFile(file)
    sheet_names = excel_file.sheet_names
    for sheet_name in sheet_names:
        df = excel_file.parse(sheet_name)

        print(f'文件 {{file}} 中工作表 {{sheet_name}} 的基本信息：')
        df.info()

        # 查看数据集行数和列数
        rows, columns = df.shape

        if rows < 50 and columns < 20:
            # 短表数据（行数少于50且列数少于20）查看全量数据信息
            print(f'文件 {{file}} 中工作表 {{sheet_name}} 的全部内容信息：')
            print(df.to_csv(sep='\t', na_rep='nan'))
        else:
            # 长表数据查看数据前几行信息
            print(f'文件 {{file}} 中工作表 {{sheet_name}} 的前几行内容信息：')
            print(df.head().to_csv(sep='\t', na_rep='nan'))
```"""
    response_code += code_str
    response1 = response_text + response_code
    return response1

def get_response_2_csv_few(sample):
    if len(sample.csv_paths) > 1:
        info = f"""数据文件内包含：{sample.csv_paths}\n"""
        for i in range(len(sample.csv_paths)):
            rows_num, columns_num = sample.csv_dfs[sample.csv_paths[i]].shape
            columns = sample.info_provider.df_columns(sample.csv_dfs[sample.csv_paths[i]])
            path = sample.csv_paths[i]
            info += f"- {path} 已经加载到 df{i+1} 变量，包含 {rows_num} 行 {columns_num} 列，其中包含的列名有：{columns}\n"
    else:
        file = sample.csv_paths[0]
        info = f"""数据文件内包含1个csv文件：{file}\n"""
        rows_num, columns_num = sample.csv_dfs[file].shape
        columns = sample.info_provider.df_columns(sample.csv_dfs[file])
        info = f"数据集已加载到 df 变量，本数据集中有 {rows_num} 行 {columns_num} 列，其中包含的列名有：{columns}\n"
    info += '\n'
    info += '我已经了解了这个数据集的基本情况，接下来我尝试回答您的问题。\n\n'
    return info

def get_response_2_csv_many(sample):
    info = f"""数据文件内包含：{sample.csv_paths}\n"""
    for file in sample.csv_paths:
        rows_num, columns_num = sample.csv_dfs[file].shape
        columns = sample.info_provider.df_columns(sample.csv_dfs[file])
        info += f"""- 文件 {file} 包含 {rows_num} 行 {columns_num} 列，其中包含的列名有：{columns}\n"""
    info += '\n'
    info += '我已经了解了这个数据集的基本情况，接下来我尝试回答您的问题。\n\n'
    return info
    
def get_response_2_excel_few(sample):
    file = sample.excel_paths[0]
    info = f"""数据文件内包含1个excel文件：{file}\n"""
    info += f"""文件 {file} 包含的工作表有：{list(sample.excel_dfs[file].keys())}\n"""
    for sheet_name in sample.excel_dfs[file]:
        rows_num, columns_num = sample.excel_dfs[file][sheet_name].shape
        columns = sample.info_provider.df_columns(sample.excel_dfs[file][sheet_name])
        info += f"""- 工作表 {sheet_name} 包含 {rows_num} 行 {columns_num} 列，包含的列名有：{columns}\n"""
    info += '\n'
    info += '我已经了解了这个数据集的基本情况，接下来我尝试回答您的问题。\n\n'
    return info
    
def get_response_2_excel_many(sample):
    info = f"""数据文件内包含：{sample.excel_paths}\n\n"""
    for file in sample.excel_paths:
        info += f"""- 文件 {file} 包含的工作表有：{list(sample.excel_dfs[file].keys())}\n"""
        for sheet_name in sample.excel_dfs[file]:
            rows_num, columns_num = sample.excel_dfs[file][sheet_name].shape
            columns = sample.info_provider.df_columns(sample.excel_dfs[file][sheet_name])
            info += f"""    - 工作表 {sheet_name} 包含 {rows_num} 行 {columns_num} 列，包含的列名有：{columns}\n"""
        info += '\n'
    info += '我已经了解了这个数据集的基本情况，接下来我尝试回答您的问题。\n\n'
    return info
    
def get_response_2_many(sample):
    info = f"""数据文件内包含的csv文件有：{sample.csv_paths}\n数据文件内包含的excel文件有：{sample.excel_paths}\n\n"""
    info += "csv文件信息如下：\n"
    for file in sample.csv_paths:
        rows_num, columns_num = sample.csv_dfs[file].shape
        columns = sample.info_provider.df_columns(sample.csv_dfs[file])
        info += f"""- 文件 {file} 包含 {rows_num} 行 {columns_num} 列，其中包含的列名有：{columns}\n"""
    info += '\n'
    info += "excel文件信息如下：\n"
    for file in sample.excel_paths:
        info += f"""- 文件 {file} 包含的工作表有：{list(sample.excel_dfs[file].keys())}\n"""
        for sheet_name in sample.excel_dfs[file]:
            rows_num, columns_num = sample.excel_dfs[file][sheet_name].shape
            columns = sample.info_provider.df_columns(sample.excel_dfs[file][sheet_name])
            info += f"""    - 工作表 {sheet_name} 包含 {rows_num} 行 {columns_num} 列，包含的列名有：{columns}\n"""
        info += '\n'
    info += '我已经了解了这个数据集的基本情况，接下来我尝试回答您的问题。\n\n'
    return info

def get_response_1(sample):
    if len(sample.csv_paths) <= sample.few_csv and len(sample.excel_paths) == 0:
        response_1 = get_response_1_csv_few(sample)
    elif len(sample.csv_paths) > sample.few_csv and len(sample.excel_paths) == 0:
        response_1 = get_response_1_csv_many(sample)
    elif len(sample.csv_paths) == 0 and len(sample.excel_paths) == 1:
        response_1 = get_response_1_excel_few(sample)
    elif len(sample.csv_paths) == 0 and len(sample.excel_paths) > 1:
        response_1 = get_response_1_excel_many(sample)
    else:
        response_1 = get_response_1_many(sample)
    return response_1

def get_response_2(sample):
    if len(sample.csv_paths) <= sample.few_csv and len(sample.excel_paths) == 0:
        response_2 = get_response_2_csv_few(sample)
    elif len(sample.csv_paths) > sample.few_csv and len(sample.excel_paths) == 0:
        response_2 = get_response_2_csv_many(sample)
    elif len(sample.csv_paths) == 0 and len(sample.excel_paths) == 1:
        response_2 = get_response_2_excel_few(sample)
    elif len(sample.csv_paths) == 0 and len(sample.excel_paths) > 1:
        response_2 = get_response_2_excel_many(sample)
    else:
        response_2 = get_response_2_many(sample)
    return response_2
