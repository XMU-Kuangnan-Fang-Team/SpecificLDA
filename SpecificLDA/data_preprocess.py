import pandas as pd
import jieba
import re
from pathlib import Path
from SpecificLDA.simulation_data import read_simulation_data


def data_preprocess(file_path = None, user_dict_path = None, stopwords_path = None):
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent.parent
    
    # 读入文本数据
    if file_path == None:
        data_dict = read_simulation_data()
    else:
        data_dict = {}
        for file in Path(file_path).iterdir():
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
    
    # 加载自定义词典(要符合jieba自定义词典规范),分为内置,不使用和自定义
    if user_dict_path == 'built':
        data_path = current_dir / 'data' /  'segmentation.txt'
        jieba.load_userdict(str(data_path))
        
    elif user_dict_path == None:
        pass
    
    else:
        jieba.load_userdict(user_dict_path)
    
    # 加载停用词库,分为内置和自定义
    if stopwords_path == 'built':
        data_path = current_dir / 'data' /  'stopwords.txt'
        with open(data_path, 'r', encoding='utf-8') as f:
            data_stopwords = f.readlines()
        for i in range(len(data_stopwords)):
            data_stopwords[i] = data_stopwords[i].replace(" ", "").replace("\n", "")

    else:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            data_stopwords = f.readlines()
    
    # 进行分词,去除停用词
    data_process = pd.DataFrame(columns=['date', 'title', 'segmentation'])
    for date, text_list in data_dict.items():
        # 提取前三个元素合并为标题，并删除连续空格
        title = ' '.join(text_list[:3])
        title = re.sub(' +', ' ', title)
        segmentation_list = []
        # 从第四个元素开始遍历，进行分词和去停用词
        for text in text_list[3:]:
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\u2003', '', text)
            text = re.sub(r'\n', '', text)
            text = re.sub(r'\xa0', '', text)
            words = jieba.lcut(text)
            words = [word for word in words if word not in data_stopwords]
            segmentation_list.append('++'.join(words))
        # 将分词结果用++连接
        segmentation = '++'.join(segmentation_list)
        # 将结果添加到DataFrame
        data_process = data_process.append({
            'date': date[:8],  # 取日期的前8位
            'title': title,
            'segmentation': segmentation
        }, ignore_index=True)
    
    return data_process

