import jieba
import numpy as np
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('load model...')
NLPmodel = load_model('C:/Users/Administrator/sentiment/senti/myUtils/sentiment.h5')
print('load done.')

print('test model...')


def generate_id2wec(model_path):
    """
       :param word2vec_model: 词向量模型位置
       :return: dictionary文字编号填入w2id，二维列表词向量embedding_weights
    """
    model = Word2Vec.load(model_path)
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model[word] for word in w2id.keys()}  # 词语的词向量
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2id,embedding_weights


w2id,embedding_weights = generate_id2wec("C:/Users/Administrator/sentiment/senti/myUtils/word2Vec.model")

# Create your models here.

class NLP():
    def __init__(self):
        model = NLPmodel

    """
    :argument
    sentiment:输入的句子
    :return:情感极性
    """
    def preSen(self, new_sen):

        new_sen_list = jieba.lcut(new_sen)
        sen2id = [w2id.get(word, 0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], 200)
        res = NLPmodel.predict(sen_input)[0]
        return np.argmax(res)
