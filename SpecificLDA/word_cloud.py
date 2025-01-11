import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from Pillow import Image
import numpy as np
from pathlib import Path


# 接收一个Dictionary对象，绘制词云图
def word_cloud(dictionary, background_color='white', image_path = None, 
              font_path = 'C:\Windows\Fonts\simhei.ttf', max_words = 100,
              stopwords = {}, max_font_size = 150, 
              random_state = 1, scale = 1):
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent.parent
    fre = {}
    for kw in list(dictionary.token2id.keys()):
        fre[kw] = dictionary.cfs[dictionary.token2id[kw]]
    
    # 背景路径为内置或给定
    if image_path == None:
        backgroud_Image = np.array(Image.open(current_dir / 'wordcloud_image.jpg'))
    else:
        backgroud_Image = np.array(Image.open(image_path))
    
    # 绘制词云图
    wc = WordCloud(
    background_color = background_color,  # 设置背景颜色，与图片的背景色相关
    mask = backgroud_Image,  # 设置背景图片
    font_path = font_path,  # 显示中文，可以更换字体
    max_words = max_words,  # 设置最大显示的字数
    stopwords = stopwords,  # 设置停用词，停用词则不再词云图中表示
    max_font_size = max_font_size,  # 设置字体最大值
    random_state = random_state,  # 设置有多少种随机生成状态，即有多少种配色方案
    scale = scale  # 设置生成的词云图的大小
    )
    
    # 传入需画词云图的文本
    wc.generate_from_frequencies(fre)
    image_colors = ImageColorGenerator(backgroud_Image)
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.show()

