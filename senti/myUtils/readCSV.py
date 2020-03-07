
def read_data(data_path):
    """
    从数据地址读取csb格式数据集
    :param data_path:数据集地址
    :return: 句子列表senlsit，标签列表labellist
    """
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
