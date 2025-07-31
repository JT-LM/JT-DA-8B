import pandas as pd


class PreProcessor:
    def __init__(self,
                 cell_max_len=50, # 单元格最大容纳长度
                 ):
        self.cell_max_len = cell_max_len

    def strip_columns(self, df):
        # 列名处理
        try:
            # 去掉列名中的前后空格和换行符
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f'去掉列名中的前后空格和换行符报错：{e}, 数据：{df}')
        return df

    def fillna(self, df):
        # 空值处理
        try: 
            for column in df.columns:
                if df[column].isnull().all():
                    # 如果整列为空，填充为 'nan'
                    df[column].fillna('nan', inplace=True)
                elif pd.api.types.is_numeric_dtype(df[column]):
                    # 如果列是数值类型（int 或 float），填充为 0
                    df[column].fillna(0, inplace=True)
                elif pd.api.types.is_string_dtype(df[column]):
                    # 如果列是字符串类型，填充为 'nan'
                    df[column].fillna('nan', inplace=True)
        except Exception as e:
            print(f'空值处理报错：{e}，数据：{df}')
        return df
    
    def cut_content(self, df):
        # 遍历每个单元格进行处理，处理过长文本，防止超上下文
        try: 
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > self.cell_max_len else x)
        except Exception as e:
            pass
        return df

