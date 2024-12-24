import pandas as pd
from pathlib import Path


def read_simulation_data(data_folder = 'newspaper'):
    
    # 获取当前文件所在的目录
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent.parent
    
    # 读入文本数据
    data_dict = {}
    data_path = current_dir / 'data' / data_folder
    # 遍历文件夹内的所有文件
    for file in data_path.iterdir():
        if file.is_file():
            file_stem = file.stem
            if file.suffix == '.csv':
                # 对于 CSV 文件，使用 pandas 读取
                data_dict[file_stem] = pd.read_csv(file)
            elif file.suffix == '.txt':
                # 对于 TXT 文件，尝试使用 UTF-8 编码读取
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data_dict[file_stem] = f.readlines()
                except UnicodeDecodeError:
                    # 如果 UTF-8 解码失败，尝试使用 GBK 解码
                    with open(file, 'r', encoding='gbk') as f:
                        data_dict[file_stem] = f.readlines()
            else:
                print(f"Skipping file with unsupported extension: {file.name}")

    return data_dict

