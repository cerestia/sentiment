from gensim.corpora.dictionary import Dictionary
from gensim.models import word2vec
import numpy as np
import json
import pandas
# 读取已经处理过的词向量模型
# 将模型转化为数值

# 读取模型
# 将词向量权重填入embedding_weights
# 将词编号填入w2id
def generate_id2wec(word2vec_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    # 词语的索引，从1开始编号
    w2id = {v: k + 1 for k, v in gensim_dict.items()}
    print(w2id)
    # 词语的词向量
    w2vec = {word: model[word] for word in w2id.keys()}
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    # 从索引为1的词语开始，用词向量填充矩阵
    for w, index in w2id.items():
        embedding_weights[index, :] = w2vec[w]
    print(embedding_weights)
    return w2id, embedding_weights


# 文本转为索引数字模式
def text_to_array(w2index, senlist):
    sentences_array = []
    for sen in senlist:
        new_sen = [w2index.get(word, 0) for word in sen]  # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)


'''
def prepare_data(w2id,sentences,labels,max_len=200):
    X_train, X_val, y_train, y_val = train_test_split(sentences,labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), np_utils.to_categorical(y_train) ,np.array(X_val), np_utils.to_categorical(y_val)
'''
model = word2vec.Word2Vec.load('word2Vec.model')
w2id, embedding_weights = generate_id2wec(model)
js = json.dumps(w2id)
with open('model.json','w') as json_file:
    json_file.write(js)
embedding_weights_temp = embedding_weights[0:100]
np.savetxt("weights.txt", embedding_weights_temp,fmt='%f',delimiter=',')
weightCSV = pandas.DataFrame(embedding_weights)
weightCSV.to_csv('weight.csv')
