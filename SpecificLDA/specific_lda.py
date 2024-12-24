import pandas as pd
from gensim.corpora.dictionary import Dictionary
from SpecificLDA.gibbs import LDAGibbsP
from SpecificLDA.data_preprocess import data_preprocess
from SpecificLDA.word_cloud import word_cloud


def specific_lda(file_path = None, user_dict_path = 'built', stopwords_path = 'built', n_rounds = 5,
                 n_iter = 5, K = 2, topic_name = 'macro_economy', priori_word = ['宏观经济','数字经济'],
                 priori_weight = 'average', priori_word_percent = 0.2, priori_data_balance = 1,
                 n_evaluate_words = 100, priori_type = 'multi_alpha', init_base = 'alpha',
                 n_top_words = 100, wordcloud = True, process_result = False):
    
    # 定义先验信息
    priori_param = {
        'topic_name' : topic_name,
        'priori_word' : priori_word, # 重要
        'priori_weight' : priori_weight, # 重要 fre/average/[w1,w2,...]
        }
    # priori_weight = [1/len(priori_word)]*len(priori_word)  #'average'
    # priori_weight = [0.5,0.25,0.125,0.125]

    # 定义模型超参数
    super_param = {
        'priori_word_percent':priori_word_percent, # 重要
        'priori_data_balance':priori_data_balance,
        'n_evaluate_words':n_evaluate_words,
        'priori_type':priori_type, # 重要 same_alpha/multi_alpha
        'init_base':init_base,
        'n_top_words':n_top_words,
    }
    
    # 分词并去除停用词
    data_process = data_preprocess(file_path, user_dict_path, stopwords_path)
    data = data_process

    # 获取文档
    document = []
    for seg in data['segmentation']:
        segs = seg.split('++')
        document.append(segs)
    data['doc'] = document
    dictionary = Dictionary(document)
    
    # 绘制词云图
    if wordcloud == True:
        word_cloud(dictionary)
    else:
        pass
    
    # 获取document id和日期
    document_idx = [list(filter(lambda x:x!=-10,dictionary.doc2idx(doc, unknown_word_index=-10))) for doc in document]
    doc_idx_len = []
    for idx,i in enumerate(document_idx):
        doc_idx_len.append(len(i))
    data['doc_idx_len'] = doc_idx_len
    date_idx = pd.to_datetime(data['date'])
    
    # 建立共现矩阵
    corpus = [dictionary.doc2bow(doc) for doc in document]
    index = [i for i in range(len(dictionary.token2id.keys()))]
    co_occur = pd.DataFrame(data = 0,index = index,columns = index)
    for doc_idx in range(len(corpus)):
        if corpus[doc_idx] == []:
            print(doc_idx)
            continue
        words_in_doc = list(pd.DataFrame(corpus[doc_idx])[0])
        for word_idx in words_in_doc:
            co_occur[word_idx][words_in_doc] += 1

    # 初始化gibbs抽样类
    plda = LDAGibbsP(dictionary=dictionary,date_idx=date_idx, 
                     cooccur = co_occur, priori_param=priori_param, 
                     super_param=super_param, K = 2)
    
    # 计算先验
    plda.get_beta()
    if super_param['priori_type'] == 'multi_alpha':
        plda.get_alpha(document_idx)
    elif super_param['priori_type'] == 'same_alpha':
        plda.get_same_alpha(document_idx)
    
    # 初始化参数
    plda._init_prior()
    plda._init_params(document_idx)
    plda._update_params()
    plda.evaluation()
    if process_result == True:
        plda.result_print()
    else:
        pass
    
    #开始迭代更新参数
    for round in range(n_rounds):
        plda.collapse_gibbs_sampling(texts=document_idx, max_iter=n_iter)
        plda._update_params()
        plda.evaluation()
        if process_result == True:
            plda.result_print()
        else:
            pass
        print('\n\n')
    params = plda.params
    params['date'] = data['date']
    return params

