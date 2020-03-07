import jieba
from gensim.corpora import Dictionary
from gensim.models.word2vec import Word2Vec
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, Activation, Softmax
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import Sequential

# 读取w2vec生成的模型
# 生成lstm模型
from .readCSV import read_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

"""
sentences,labels = read_data("data_train.csv")

w2id,embedding_weights = generate_id2wec("word2Vec.model")

for id,w in w2id.items():
    print(id,w)
"""
def text_to_array(w2index, senlist):  # 文本转为索引数字模式
    """
       文本转为索引数字模式
       :param w2index: 词索引
       :param senlist: 句子列表
       :return: 词编号列表
       """
    sentences_array = []
    for sen in senlist:
        new_sen = [ w2index.get(word,0) for word in sen]   # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)

def prepare_data(w2id,sentences,labels,max_len=200):
    """
        数据预处理，将分出训练集，测试集，并做序列预处理
        :param w2id: 词编号词典
        :param sentences: 句子列表
        :param labels: 标签列表
        :param max_len:
        :return:训练集，测试集
        """
    X_train, X_val, y_train, y_val = train_test_split(sentences,labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), np_utils.to_categorical(y_train) ,np.array(X_val), np_utils.to_categorical(y_val)

"""
x_train,y_trian, x_val , y_val = prepare_data(w2id,sentences,labels,200)
"""

class Sentiment:
    def __init__(self, w2id, embedding_weights, Embedding_dim, maxlen, labels_category):
        self.Embedding_dim = Embedding_dim
        self.embedding_weights = embedding_weights
        self.vocab = w2id
        self.labels_category = labels_category
        self.maxlen = maxlen
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        # input dim(140,100)
        model.add(Embedding(output_dim=self.Embedding_dim,
                            input_dim=len(self.vocab) + 1,
                            weights=[self.embedding_weights],
                            input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.labels_category))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train, X_test, y_test, n_epoch=1):
        self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch,
                       validation_data=(X_test, y_test))
        self.model.save('sentiment.h5')

    def predict(self, new_sen):
        model = self.model
        #model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id = [self.vocab.get(word, 0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        return np.argmax(res)

"""
senti = Sentiment(w2id, embedding_weights, 100, 200, 3)
    senti.train(x_train,y_trian, x_val ,y_val,1)
if __name__ == "__main__":
"""