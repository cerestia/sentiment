import jieba
import numpy as np
from gensim.models import Word2Vec
from readCSV import read_data

# 该模块负责生成词向量模型
# 保存在文件夹中
"""
def read_data(data_path):
    
    从数据地址读取csb格式数据集
    :param data_path:数据集地址
    :return: 句子列表senlsit，标签列表labellist
   
    senlist = []
    labellist = []
    with open(data_path, "r", encoding='gb2312', errors='ignore') as f:
        for data in f.readlines():
            data = data.strip()
            sen = data.split("\t")[2]
            label = data.split("\t")[3]
            if sen != "" and (label == "0" or label == "1" or label == "2"):
                senlist.append(sen)
                labellist.append(label)
            else:
                pass
    assert (len(senlist) == len(labellist))
    return senlist, labellist
"""

sentences, labels = read_data("data_train.csv")


def train_word2vec(sentences, save_path):
    """
    训练词向量
    :param sentences: 评论列表
    :param save_path: 保存位置
    :return: 词向量模型
    """
    sentences_seg = []
    sen_str = "\n".join(sentences)
    res = jieba.lcut(sen_str)
    seg_str = " ".join(res)
    sen_list = seg_str.split("\n")
    for i in sen_list:
        sentences_seg.append(i.split())
    #     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences_seg,
                     size=100,  # 词向量维度
                     min_count=5,  # 词频阈值
                     window=5)  # 窗口大小
    model.save(save_path)
    return model

"""
if __name__ == "__main__":
    print("开始训练词向量")
    model = train_word2vec(sentences, 'word2Vec.model')
    print("训练完成")
"""