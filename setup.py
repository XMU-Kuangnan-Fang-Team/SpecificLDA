from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SpecificLDA',
    version='0.0.1',
    description='A Python library implementing the Specific LDA model for targeted extraction of specific topics from text data',
    author='Kuangnan Fang, Hongwei Lin, Yuhao Zhong',
    author_email='XMUFKN@163.com',
    url='https://github.com/ruiqwy/SpecificLDA',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'wordcloud',
        'PIL',
        'pathlib',
        'jieba',
        'gensim',
        'tqdm',
        'setuptools' 
    ],
    inlude_package_data = True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
    ],
)