## `SpecificLDA`: 一个用于从文本数据中针对性提取特定主题的Python包

### 介绍
SpecificLDA是一个基于定向LDA模型，用于从文本中提取指定主题信息，并构建注意力指数的Python包。该库提供了文本预处理、模型构建和趋势分析等功能，在实际应用中，可以用于从官方媒体新闻文本中提取与国家治理现代化相关的主题信息，并构建国家治理政府注意力指数。若要使用该Python包并进行研究分析，请引用文章：

*方匡南 戴明晓 郑挺国 林洪伟. 国家治理政府注意力指数构建及其应用—基于新闻文本的测度. 统计研究. 2025*

具体的方法与数据分析内容也请参照以上文章。

### 安装
推荐从Github上下载并安装该Python包：
```c
git clone https://github.com/XMU-Kuangnan-Fang-Team/SpecificLDA.git
cd SpecificLDA
```
之后运行以下代码来安装项目：
```c
pip install .
```

### 主要函数
#### 目录
- [read_simulation_data](#read_simulation_data)
- [data_preprocess](#data_preprocess)
- [specific_lda](#specific_lda)
- [epu_plot](#epu_plot)

#### read_simulation_data
用于读取示例文本数据的函数。该函数仅读取内置的示例数据，返回一个键为日期值为文本的字典。参数不需要设置。
##### 用法
```c
read_simulation_data(data_folder = 'newspaper')
```
##### 参数
|参数|描述|
|:---:|:---:|
data_folder|示例数据的路径。
##### 示例
```c
newspaper_data = read_simulation_data()
```

#### data_preprocess
用于对文本数据进行分词和去除停用词操作的函数。输入为待处理的文档路径，分词表路径和停用词路径，输出为一个包含日期和分隔文本的矩阵。
##### 用法
```c
data_preprocess(file_path = None, user_dict_path = 'built', stopwords_path = 'built')
```
##### 参数
|参数|描述|
|:---:|:---:|
file_path|待处理文档的路径。输入形式应为一个包含多个txt或csv文件的文件夹，读取该文件夹的路径。若为None则读取内置示例数据。
user_dict_path|分词表的路径。分为内置(built)，不使用(None)和自定义，格式为词汇，权重(可选)和词性(可选)。
stopwords_path|停用词文档的路径。分为内置(built)和自定义，格式为一行一个停用词。
##### 示例
```c
process_data = data_preprocess(file_path = None, user_dict_path = 'built', stopwords_path = 'built')
```

#### specific_lda
用于运行定向LDA模型的函数，返回估计出的参数，分类结果以及每个文本的日期。
##### 用法
```c
specific_lda(data_process, n_rounds = 5, n_iter = 5, K = 2, topic_name = 'macro_economy',
             priori_word = ['宏观经济','数字经济'],
             priori_weight = 'average', priori_word_percent = 0.2, priori_data_balance = 1,
             n_evaluate_words = 100, priori_type = 'multi_alpha', init_base = 'alpha',
             n_top_words = 100, wordcloud = True, process_result = False)
```
##### 参数
|参数|描述|
|:---:|:---:|
data_process|由data_reprocess函数处理后的文本。
n_rounds|循环执行的轮数。
n_iter|循环迭代的次数。
K|文本分类的主题数。
topic_name|文本的主题名。
priori_word|先验词，可以有一个或多个，用列表储存。
priori_weight|文本词语的先验权重，可以为fre(基于频数)或average(平均)或自定义。自定义权重由列表储存。
priori_word_percent|先验词的百分比。
priori_data_balance|数据平衡参数。
n_evaluate_words|评估词的数目。
priori_type|先验类型，same_alpha表示所有文档使用相同的先验分布，multi_alpha表示允许不同的先验分布。
init_base|基于alpha初始化(alpha)或基于beta初始化(beta)。
n_top_words|顶级词的数量，确定主题模型中每个主题要提取的顶级词的数量。
wordcloud|是否绘制词云图。
process_result|是否输出过程的建模结果。
##### 示例
```c
params = specific_lda(file_path,priori_word = ['宏观经济','数字经济'],
                      priori_weight = 'average', priori_word_percent = 0.2, priori_data_balance = 1,
                      n_evaluate_words = 100, priori_type = 'same_alpha', init_base = 'alpha',
                      n_top_words = 100)
```

#### epu_plot
用于计算注意力指数和绘制注意力指数随时间变化趋势图的函数。该函数读取由specific_lda函数生成的参数结果，返回一个键为日期值为文本的字典并作图。
##### 用法
```c
epu_plot(params, roll, figsize = (10,4))
```
##### 参数
|参数|描述|
|:---:|:---:|
params|由specific_lda函数生成的参数结果。
roll|滚动平均窗口的大小。
figsize|生成图片的画布比例。
##### 示例
```c
params = specific_lda(file_path,priori_word = ['宏观经济','数字经济'],
                      priori_weight = 'average', priori_word_percent = 0.2, priori_data_balance = 1,
                      n_evaluate_words = 100, priori_type = 'same_alpha', init_base = 'alpha',
                      n_top_words = 100)
epu_day = epu_plot(params, 2)
```
