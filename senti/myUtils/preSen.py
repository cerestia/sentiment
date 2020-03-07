from .readVec import senti


def pre(sen_new):
    label_dic = {0: "消极的", 1: "中性的", 2: "积极的"}
    pre = senti.predict("./sentiment.h5", sen_new)
    #print("'{}'的情感是:\n{}".format(sen_new, label_dic.get(pre)))
    return label_dic.get(pre)


#print(pre("垃圾分类做的还可以"))
