import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math


class LDAGibbsP(object):
    def __init__(self, dictionary, date_idx, cooccur, priori_param, super_param, K = 2):
        """
        利用collapsed gibbs sampling方法实现LDA
        :param K: 主题个数，默认值为2
        :param V: 词汇表总共单词个数
        :param M: 文本总数
        :param tokens: 单词tokens
        :param alpha: theta的超参数，维度(K,)
        :param theta: 文本主题矩阵，服从狄利克雷分布，维度(M,K)
        :param beta: varphi的超参数，维度(V,)
        :param varphi: 单词主题矩阵，服从狄利克雷分布，维度(K,V)
        :param z: 话题集合，维度(M,Nm),Nm表示第m个文本的单词个数
        """
        self.K = K
        self.dictionary = dictionary
        self.tokens = list(dictionary.values())
        self.priori_param = priori_param
        self.super_param = super_param
        self.cooccur = cooccur
        self.date_idx = date_idx
        self.fre = {}  # token:频数
        for kw in self.tokens:
            self.fre[kw] = self.dictionary.cfs[self.dictionary.token2id[kw]]
        self.n_words = np.sum(list(self.fre.values()))
        print('过滤之后，文章包含总词数:', self.n_words)
        self.n_kw_words = 0
        for p in self.priori_param['priori_word']:
            self.n_kw_words += self.fre[p]
        print('先验词词频总数：', self.n_kw_words)
        self.kw_ratio = self.n_kw_words / self.n_words
        print('先验词词频占比', self.kw_ratio)
        self.V = len(self.tokens)
        self.M = None
        self.n_iter = 0
        self.n_update_iter = []
        self.ratio_update_iter = []
        self.coherence = []
        self.params = {
            'alpha': None,
            'theta': None,
            'beta': None,
            'varphi': None,
            'z': None
        }


    def _init_prior(self,):
        '''
        初始化参数，包括alpha和beta
        :return:
        '''
        # self.M = len(texts)
        # self.V = len(self.tokens)
        # self.n_words = sum(self.dictionary.cfs.values())
        # 仅仅起到占位作用
        # 文档主题参数先占位，最后再进行更新
        theta = np.ones((self.M, self.K))
        # 单词主题参数也先占位，最后再进行更新
        varphi = np.ones((self.K, self.V))
        # if type(self.alpha) != np.ndarray:
        #     self.alpha = np.ones((self.M, self.K))
        # if type(self.beta) != np.ndarray:
        #     self.beta = np.ones((self.K, self.V))
        self.params = {
            'alpha': self.alpha,
            'beta': self.eta,
            'theta': theta,
            'varphi': varphi,
        }


    def _init_params(self, texts):
        """
        初始化参数，除了上面定义的参数，公式中统计的参数也在这儿初始化
        :param texts: 输入文本
        :return:
        """
        # 记录主题标志，初始化
        z = []
        for m in range(self.M):
            N = len(texts[m])
            z.append([0]*N)
        # z = np.array(z)
        # 代表第k个主题的第v个单词个数
        self.n_kv = np.zeros((self.K, self.V))
        # 代表第k个主题有多少个单词
        self.n_k = np.zeros(self.K)
        # 代表第m个文档第k个主题的单词个数
        self.n_mk = np.zeros((self.M, self.K))
        # 每一个元素代表第m个文档有多少个单词
        self.n_m = np.zeros(self.M)
        # 初始化中做一次统计，且对z随机初始化
        for m in tqdm(range(self.M)):  # 第m篇文档
            N = len(texts[m])
            for v in range(N):  # 第m篇文档的第v个词
                if self.super_param['init_base'] == 'uniform':
                    rand_topic = int(np.random.randint(0, self.K))  # 均匀在K个主题中抽取
                elif self.super_param['init_base'] == 'alpha':
                    p = self.params['alpha'][m] / sum(self.params['alpha'][m])
                    rand_topic = np.argmax(np.random.multinomial(1, p))  # 根据先验alpha在K个主题中抽取
                elif self.super_param['init_base'] == 'beta':
                    p = self.params['beta'][:, int(texts[m][v])] / sum(self.params['beta'][:, int(texts[m][v])])
                    rand_topic = np.argmax(np.random.multinomial(1, p))  # 根据先验alpha在K个主题中抽取
                z[m][v] = int(rand_topic)
                self.n_kv[rand_topic][int(texts[m][v])] += 1
                self.n_k[rand_topic] += 1
                self.n_mk[m][rand_topic] += 1
            self.n_m[m] = N
        self.params['z'] = z


    def _sample_topic(self, m, v, texts):
        """
        计算z_mv对应主题的概率值，并通过概率值进行多项分布采样，得到当前的主题
        math:
        p(z):
        $$p(z_{mv} | z_{-mv}, w , alpha, beta) =  frac {n_{kv} +beta_v -1} {sum_{v=1}^V (n_{kv} + beta_v) -1} .
        frac {n_{mk} + alpha_k - 1} {sum_{k=1}^K (n_{mk} + alpha_k) - 1}$$

        :param m: 表示调用的时候，上层函数计算的是第m个文档
        :param v: 第v个单词
        :param texts: 语料
        :return:
        """
        # 首先是排除当前的单词z[m][v]
        old_topic = int(self.params['z'][m][v])
        self.n_kv[old_topic][int(texts[m][v])] -= 1
        self.n_k[old_topic] -= 1
        self.n_mk[m][old_topic] -= 1
        self.n_m[m] -= 1
        # 依次计算该单词的p(z_mv=k | *)
        p = np.zeros(self.K)
        for k in range(self.K):
            #             p[k] = (self.n_mk[m][k] + self.params['alpha'][k]) / (self.n_k[k] + np.sum(self.params['beta'])) * \
            #                    (self.n_kv[k][int(texts[m][v])] + self.params['beta'][int(texts[m][v])]) \
            #                    / (self.n_k[k] + np.sum(self.params['beta']))
            p[k] = (self.n_mk[m][k] + self.params['alpha'][m][k]) / (self.n_m[m] + np.sum(self.params['alpha'][m])) * \
                   (self.n_kv[k][int(texts[m][v])] + self.params['beta'][k][int(texts[m][v])]) \
                   / (self.n_k[k] + np.sum(self.params['beta'][k]))
        # 对概率进行归一化处理
        p = p / np.sum(p)
        # 抽样新的主题
        new_topic = np.argmax(np.random.multinomial(1, p))
        # 更新统计值
        self.n_kv[new_topic][int(texts[m][v])] += 1
        self.n_k[new_topic] += 1
        self.n_mk[m][new_topic] += 1
        self.n_m[m] += 1
        return new_topic


    def collapse_gibbs_sampling(self, texts, max_iter=1):
        """
        吉布斯采样的循环入口
        :param texts: 语料
        :param max_iter: 最大循环次数
        :return:
        """
        for iter in range(max_iter):
            n_update = 0
            #             print('iter: {} total_iter: {}'.format(iter + 1,self.n_iter+1))
            for m in tqdm(range(self.M)):
                N = len(texts[m])
                for v in range(N):
                    old_topic = int(self.params['z'][m][v])
                    topic = self._sample_topic(m, v, texts)
                    self.params['z'][m][v] = int(topic)
                    if topic != old_topic:
                        n_update += 1
            self.n_iter += 1
            self.n_update_iter.append(n_update)
            self.ratio_update_iter.append(n_update/self.n_words)
            print('update {} {} word topics in iter {} and total_iter {} '.format(n_update, n_update/self.n_words, iter + 1, self.n_iter))


    def _update_params(self):
        """
        更新参数
        math:
        theta:
        $$theta_{mk} = frac {n_{mk} + alpha_k - 1} {sum_{k=1}^K (n_{mk} + alpha_k) - 1}$$
        varphi:
        $$varphi_{kv} = frac {n_{kv} + beta_v -1} {sum_{v=1}^V (n_{kv} + beta_v) -1} $$

        但是一般都没有减一，是因为在统计过程中，提前减一了
        :return:
        """
        # 依据统计值，更新单词主题矩阵和文档主题矩阵
        for k in range(self.K):
            for v in range(self.V):
                self.params['varphi'][k][v] = \
                    (self.n_kv[k][v] + self.params['beta'][k][v]) / \
                    (self.n_k[k] + np.sum(self.params['beta'][k]))
                # !!!与词频相关
        for m in range(self.M):
            for k in range(self.K):
                self.params['theta'][m][k] = \
                    (self.n_mk[m][k] + self.params['alpha'][m][k]) / \
                    (self.n_m[m] + np.sum(self.params['alpha'][m]))


    def get_beta(self,):
        self.rd = (1 - self.super_param['priori_word_percent']) /\
             (len(self.dictionary.token2id.keys()) - len(self.priori_param['priori_word']))
        print('其他词先验占比rd: ', self.rd)
        eta_i = np.full((len(self.dictionary.token2id.keys())), self.rd)
        self.priori_word_idx = []
        self.priori_word_fre = []
        print('序号\t', '先验词\t', '位置\t', '词频\t', '权重')
        for i, p in enumerate(self.priori_param['priori_word']):
            idx = self.dictionary.token2id[p]
            num = self.fre[p]
            w = num / self.n_kw_words
            print(i + 1, '\t', p, '\t', idx, '\t', num, '\t', w)
            self.priori_word_fre.append(w)
            self.priori_word_idx.append(idx)
        if self.priori_param['priori_weight'] == 'fre':
            self.priori_weight_used = self.priori_word_fre[:]
        elif self.priori_param['priori_weight'] == 'average':
            self.priori_weight_used = [1/len(self.priori_param['priori_word'])]*len(self.priori_param['priori_word'])
        else:
            self.priori_weight_used = self.priori_param['priori_weight'][:]
        for i, idx in enumerate(self.priori_word_idx):
            eta_i[idx] = self.super_param['priori_word_percent'] * self.priori_weight_used[i]
        # 其余混杂主题的词分布先验，1-eta_i得到相反的词语权重大小关系，再进行归一化处理
        eta_j = (1 - eta_i) / np.sum(1 - eta_i)
        self.eta = np.array([eta_i * self.n_kw_words / self.super_param['priori_word_percent'],
                        eta_j * (self.n_words - self.n_kw_words / self.super_param['priori_word_percent'])])


    def get_alpha(self, document):
        self.M = len(document)
        self.alpha = np.zeros((self.M, self.K))
        count = 0
        threshold = self.kw_ratio / self.super_param['priori_word_percent'] - self.kw_ratio
        for doc_idx, doc in enumerate(tqdm(document)):
            for w in doc:
                if w in self.priori_word_idx:
                    count += 1
                    self.alpha[doc_idx, 0] += 1
                else:
                    if np.random.rand() < threshold:
                        count += 1
                        self.alpha[doc_idx, 0] += 1
                    else:
                        self.alpha[doc_idx, 1] += 1
        print('初始化后属于关注主题的词数 {} 占比 {}'.format(count, count / sum(self.dictionary.cfs.values())))
        self.alpha = self.alpha * self.super_param['priori_data_balance']


    def get_same_alpha(self, document):
        self.M = len(document)
        self.alpha = np.zeros((self.M, self.K))
        self.alpha[:, 0] = self.kw_ratio / self.super_param['priori_word_percent']
        self.alpha[:, 1] = 1 - self.kw_ratio / self.super_param['priori_word_percent']
        # alpha[:, 0] = kw_ratio
        # alpha[:, 1] = 1-kw_ratio
        # alpha[:, 0] = param_priori_percent
        # alpha[:, 1] = 1-param_priori_percent
        words_per_doc = self.n_words / self.dictionary.num_docs
        self.alpha = self.alpha * words_per_doc
        self.alpha = self.alpha * self.super_param['priori_data_balance']


    def result_print(self,):
        for i, topic_dist in enumerate(self.params['varphi']):
            topic_words = np.array(self.tokens)[np.argsort(topic_dist)][:-(self.super_param['n_top_words'] + 1):-1]
            topic_weight = topic_dist[np.argsort(topic_dist)][:-(self.super_param['n_top_words'] + 1):-1]
            print('Topic {}: {}\n\t{}'.format(i, ' '.join(topic_words), list(topic_weight)))


    def save_results(self, dirs):
        if self.n_iter == 0:
            n_update = 0
            ratio_update = 0
        else:
            n_update = self.n_update_iter[-1]
            ratio_update = self.ratio_update_iter[-1]
        filename = 'parameters in each iter.txt'
        file_dirs = os.path.join(dirs, filename)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(file_dirs, 'a', encoding='utf-8') as f:
            f.write(
                'ITER {} :\tupdate {} {} word topics in total_iter {} \n'.format(self.n_iter, n_update, ratio_update,
                                                                                 self.n_iter))
            for i, topic_dist in enumerate(self.params['varphi']):
                topic_words = np.array(self.tokens)[np.argsort(topic_dist)][:-(self.super_param['n_top_words'] + 1):-1]
                topic_weight = topic_dist[np.argsort(topic_dist)][:-(self.super_param['n_top_words'] + 1):-1]
                f.write('\tTopic {}: {}\n\t\t{}\n'.format(i, ' '.join(topic_words), list(topic_weight)))
                # f.write('\tTopic {}: {}\n\t\t{}\n'.format(i, ' '.join(topic_words), list(topic_weight)))
            f.write('\n')


    def save_params(self, dirs):
        record = {}
        record['K'], record['V'], record['M']  = self.K, self.V, self.M
        record['kw_ratio'], record['n_words'], record['priori_weight_used'] = self.kw_ratio, self.n_words ,self.priori_weight_used
        record['n_iter'] = self.n_iter
        record['n_update'], record['ratio_update'] = self.n_update_iter, self.ratio_update_iter
        record['coherence'] = self.coherence
        record['z'] = self.params['z']
        record['n_kv'], record['n_mk'] = self.n_kv.tolist(), self.n_mk.tolist()
        record['alpha'] = self.params['alpha'].tolist()
        record['beta'] = self.params['beta'].tolist()
        record['theta'] = self.params['theta'].tolist()
        record['varphi'] = self.params['varphi'].tolist()
        record['topic_name'] = self.priori_param['topic_name']
        record['priori_word'] = self.priori_param['priori_word']
        record['priori_weight'] = self.priori_param['priori_weight']
        record['priori_word_percent'] = self.super_param['priori_word_percent']
        record['priori_data_balance'] = self.super_param['priori_data_balance']
        record['n_evaluate_words'] = self.super_param['n_evaluate_words']
        record['priori_type'] = self.super_param['priori_type']
        record['init_base'] = self.super_param['init_base']
        record['n_top_words'] = self.super_param['n_top_words']
        record['date_idx'] = list(self.date_idx)
        # self.rocord = record
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        filename = 'parameters in iter {}.csv'.format(self.n_iter)
        file_dirs = os.path.join(dirs, filename)
        record2 = {}
        for key in record.keys():
            if type(record[key]) == list:
                record2[key] = [record[key]]
            elif type(record[key]) == np.ndarray:
                record2[key] = [record[key]]
            else:
                record2[key] = record[key]
        pd.DataFrame(record2).to_csv(file_dirs)


    def evaluation(self,):
        coherences = []
        for i, topic_dist in enumerate(self.params['varphi']):
            topic_words = np.array(self.tokens)[np.argsort(topic_dist)][:-(self.super_param['n_evaluate_words'] + 1):-1]
            topic_idx = []
            for t in topic_words:
                topic_idx.append(self.dictionary.token2id[t])
            pmi = pd.DataFrame(0, columns=topic_idx, index=topic_idx)
            for x in topic_idx:
                for y in topic_idx:
                    if x == y:
                        pmi.loc[x, y] = 0
                    elif self.cooccur.loc[x, y] == 0:
                        pmi.loc[x, y] = 0
                    else:
                        value = math.log(self.cooccur.loc[x, y] / self.cooccur.loc[x, x] / self.cooccur.loc[y, y] * self.M, 2)
                        pmi.loc[x, y] = max(value, 0)
            coherences.append(np.sum(np.sum(pmi, axis=0)) / 2 / (len(topic_idx)*(len(topic_idx)-1)/2))
        print('coherence_score:\n\tTopic {}: {}\n\tTopic {}: {}'.format('1', coherences[0], '2', coherences[1]))
        self.coherence.append(coherences)


    def load(self, model_path,):
        record = pd.read_csv(model_path)
        self.K, self.V, self.M = record['K'].item(), record['V'].item(), record['M'].item()
        self.kw_ratio, self.n_words = record['kw_ratio'].item(), record['n_words'].item()
        self.priori_weight_used = eval(record['priori_weight_used'].item())
        self.n_iter = record['n_iter'].item()
        self.n_update_iter = eval(record['n_update'].item())
        self.ratio_update_iter = eval(record['ratio_update'].item())
        self.n_kv, self.n_mk = np.array(eval(record['n_kv'].item())), np.array(eval(record['n_mk'].item()))
        self.params['z'] = eval(record['z'].item())
        self.params['alpha'] = np.array(eval(record['alpha'].item()))
        self.params['beta'] = np.array(eval(record['beta'].item()))
        self.params['theta'] = np.array(eval(record['theta'].item()))
        self.params['varphi'] = np.array(eval(record['varphi'].item()))
        self.n_m, self.n_k = [], []
        for m in self.n_mk:
            self.n_m.append(sum(m))
        for k in self.n_kv:
            self.n_k.append(sum(k))
        self.n_m, self.n_k = np.array(self.n_m), np.array(self.n_k)

