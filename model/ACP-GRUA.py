import tensorflow as tf
import numpy as np
import re

from Bio import SeqIO
from matplotlib import pyplot as plt
from propy.QuasiSequenceOrder import GetSequenceOrderCouplingNumberTotal
from sklearn import metrics
from sklearn.model_selection import KFold
from propy.PseudoAAC import GetAAComposition
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
import keras.backend as K
from keras.layers import Lambda
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
import torch.nn as nn
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from Bio import SeqIO
from propy.PseudoAAC import GetAPseudoAAC
from propy.CTD import CalculateCTD
from sklearn.model_selection import train_test_split
from multiprocessing import Pool,cpu_count
import numpy as np
from sklearn import metrics
from propy.QuasiSequenceOrder import GetQuasiSequenceOrder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import minmax_scale
import propy
from modlamp.descriptors import *
import tensorflow as tf
def DDE(seq):
	AA ='ACDEFGHIKLMNPQRSTVWY'
	encodings=[]
	myCodons = {
		'X': 0,
		'A': 4,
		'C': 2,
		'D': 2,
		'E': 2,
		'F': 2,
		'G': 4,
		'H': 2,
		'I': 3,
		'K': 2,
		'L': 6,
		'M': 1,
		'N': 2,
		'P': 4,
		'Q': 2,
		'R': 6,
		'S': 6,
		'T': 4,
		'V': 4,
		'W': 1,
		'Y': 2
	}

	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

	myTM = []
	for pair in diPeptides:
		myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	#for i in fastas:
	sequence = seq
	code = []
	tmpCode = [0] * 400
	for j in range(len(sequence) - 2 + 1):
		tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
	if sum(tmpCode) != 0:
		tmpCode = [i/sum(tmpCode) for i in tmpCode]

	myTV = []
	for j in range(len(myTM)):
		myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

	for j in range(len(tmpCode)):
		tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

	code = code + tmpCode
	return code
def BPNC(seq):
    code = []
    tem =[]
    k = 7
    for i in range(k):
        if seq[i] =='A':
            tem = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='C':
            tem = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='D':
            tem = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='E':
            tem = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='F':
            tem = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='G':
            tem = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='H':
            tem = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='I':
            tem = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='K':
            tem = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='L':
            tem = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='M':
            tem = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='N':
            tem = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif seq[i]=='P':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif seq[i]=='Q':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif seq[i]=='R':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif seq[i]=='S':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif seq[i]=='T':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif seq[i]=='V':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif seq[i]=='W':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif seq[i]=='Y':
            tem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        code += tem
    return code

def CKSAAGP(fastas, gap=3):
    # 计算被任意k个残基隔开的氨基酸对的频率根据物理化学性质组成的不同组别

    def generateGroupPairs(groupKey):
        # CKSAAGP的子函数
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        return gPair

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []

    name, sequence = [], fastas
    code = [name]
    for g in range(gap + 1):
        gPair = generateGroupPairs(groupKey)
        sum = 0
        for p1 in range(len(sequence)):
            p2 = p1 + g + 1
            if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                    sequence[p2]]] + 1
                sum = sum + 1

        if sum == 0:
            for gp in gPairIndex:
                code.append(0)
        else:
            for gp in gPairIndex:
                code.append(gPair[gp] / sum)
    code = code[1:]
    return code

def GF(seq):
    # r1 = list(GetAPseudoAAC(seq, lamda=5).values())#PACC
    # x = ProteinAnalysis(seq)
    # r3 = [x.molecular_weight()]#分子量
    # r4 = list(x.get_amino_acids_percent().values())#频率
    # r5 = [x.charge_at_pH(pH=i) for i in 14]#PH
    # x6 = list(x.secondary_structure_fraction())#香农熵
    seq=str(seq)
    desc=GlobalDescriptor(seq)
    x9=desc.aliphatic_index()#脂肪族
    x1=[float(desc.descriptor)]
    x2=desc.hydrophobic_ratio()#疏水性
    x2=[float(desc.descriptor)]
    x3=desc.boman_index()#蛋白质相互作用
    x3=[float(desc.descriptor)]
    x4=desc.instability_index()#不稳定指数
    x4=[float(desc.descriptor)]
    x5=desc.isoelectric_point()#等电性
    x5=[float(desc.descriptor)]
    res = x1+x2+x3+x4+x5
    return res
def PAAC(seq):
    r1 = list(GetAPseudoAAC(seq, lamda=5).values())
    return r1
def DPC(seq: str):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    name, sequence = [], seq
    code = [name]
    tmpCode = [0] * 400
    for j in range(len(sequence) - 2 + 1):
        tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
            sequence[j + 1]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]
    code = code + tmpCode
    return code[1:]


def CPAAC(seq):  # 物理和化学性质（阳离子和双亲性）
    window = 5
    code = []
    group = {
        'Strongly hydrophilic or polar': 'RDENQKH',
        'Strongly hydrophobic': 'LIVAMF',
        'Weakly hydrophilic or weakly hydrophobic': 'STYW',
        'Proline': 'A',
        'glycine': 'L',
        'Cysteine': 'H',
    }
    myDict = {}
    groupKey = group.keys()
    for i in rx:
        name = i.id
        sequence = i.seq
        code = [name]
        for j in range(len(sequence)):
            if j + window <= len(sequence):
                count = Counter(sequence[j:j + window])
                myDict = {}
                for key in groupKey:
                    for aa in group[key]:
                        myDict[key] = myDict.get(key, 0) + count[aa]
                for key in groupKey:
                    code.append(myDict[key] / window)
        return code[1:]



def AAC(seq):  # 每个氨基酸的频率
    code = GetAAComposition(seq)
    res = []
    for v in code.values():
        res.append(v)
    return res
def SOCNumber(seq:str):
    code=GetSequenceOrderCouplingNumberTotal(seq)
    res=[]
    for v in code.values():
        res.append(v)
    return res

def encode(seq):
    ''''#x1 = BPF(seq)
    x2 = AAC(seq)

SOCNumber(seq)
    x5 = '''
    x1=GF(seq)#理化性质
    x2 = BPNC(seq)#二元图谱特征
    x3 = DPC(seq)#二肽数量
    x4 = CKSAAGP(seq)#氨基酸对的分离频率
    x5 = PAAC(seq)#伪氨基酸组成
    x6=AAC(seq)
    x7=  DDE(seq)#二肽残基
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)
    x5 = np.array(x5)
    x6=np.array(x6)
    x7=np.array(x7)
    '''x2 = np.array(x2)


    x5 = np.array(x5)'''

    res = np.concatenate([x1,x3,x4,x6,x7], axis=-1)  #,,x7,x2x4,x6,x2,x6,,x4x,x22 ,x4,x2,,x3x5x2,x7,,x4,,x6, ///134 14 13 34//x1x3:0.88,x7,x4,x2,x6,x3,,x1,x9
    #x7+x6:0.83
    return res


def calculate_specificity(y_num, y_pred, y_true):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(y_num):
        if y_true[i] == 1:
            if y_true[i] == y_pred[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_true[i] == y_pred[i]:
                tn = tn + 1
            else:
                fp = fp + 1
    specificity = float(tn) / (tn + fp)
    return specificity


batchsize = 30
epochs = 69

performance = []  # 综合表现
fp = []
tp = []
roc_auc = []

kf = KFold(n_splits=5, shuffle=True, random_state=10)  # 五折
#rx = list(SeqIO.parse("D:\ACP-check-main\datasets\ACPred-Fuse/train.txt", format="fasta"))
#rx = list(SeqIO.parse("D:\ACP-check-main\datasets\ACP-DL/acp240.txt", format="fasta"))
#rx= list(SeqIO.parse("C:/Users/21692\Desktop\ACP-check-main\datasets\ACPred-FL/all.txt", format="fasta"))
rx = list(SeqIO.parse("C:/Users/21692\Desktop\新建文件夹\项目\ACPs250.txt", format="fasta"))
protein_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
# protein_dict = {'A':0.25,'C':0.75,'D':0,'E':0,'F':0,'G':0.5,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':0,'S':0,'T':0,'V':0,'W':0,'Y':0}

for train_index, test_index in kf.split(rx):
    seqlen = []
    # 模型的第一个输入，进行embedding
    train_data1 = []
    train_label = []
    for i in train_index:
        seqlen.append(len(rx[i].seq))
        if str(rx[i].id).endswith("1"):
            train_data1.append(str(rx[i].seq))
            train_label.append(1)
        if str(rx[i].id).endswith("0"):
            train_data1.append(str(rx[i].seq))
            train_label.append(0)

    test_data1 = []
    test_label = []
    for i in test_index:
        if str(rx[i].id).endswith("1"):
            test_data1.append(str(rx[i].seq))
            test_label.append(1)
        if str(rx[i].id).endswith("0"):
            test_data1.append(str(rx[i].seq))
            test_label.append(0)

    # 数据集按照字典进行编码
    train1 = []
    for seq in train_data1:
        tmp = []
        for i in seq:
            tmp.append(protein_dict[i])
        train1.append(tmp)

    test1 = []
    for seq in test_data1:
        tmp = []
        for i in seq:
            tmp.append(protein_dict[i])
        test1.append(tmp)
    # 找出最长的序列长度
    maxlen = 0
    for i in train1:
        if len(i) >= maxlen:
            maxlen = len(i)
    for i in test1:
        if len(i) >= maxlen:
            maxlen = len(i)

    # 不够长的序列填0
    train1 = pad_sequences(train1, maxlen=maxlen, padding='post')
    test1 = pad_sequences(test1, maxlen=maxlen, padding='post')
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    # print(train1)
    # print(in_x1)
    train1 = np.array(train1)
    test1 = np.array(test1)
    # test1=np.array(test1)
    # 填充后的值矩阵
    sen = []
    sen1 = []
    sen2 = []
    for i in train1:
        sen = []
        for j in i:
            sen1 = []
            sen1.append(j)
            sen.append(sen1)
        sen2.append(sen)
    cqr = []
    for i in sen2:
        # print(i)
        word_vector = []
        model = Word2Vec(i, vector_size=900, window=3, min_count=1, workers=4)
        for j in i:
            word_vector.append(model.wv[j])
        cqr.append(word_vector)
    # 位置编码
    # import numpy as np
    import matplotlib.pyplot as plt
    import torch


    # 根据给定的最大序列长度和嵌入维度生成位置编码。
    def get_positional_encoding(max_seq_len, embed_size):
        positional_encoding = np.array([
            [pos / np.power(10000, 2 * (i // 2) / embed_size) for i in range(embed_size)]
            for pos in range(max_seq_len)])
        positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])  # 偶数索引列
        positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])  # 奇数索引列

        # 添加批次维度（Batch size）
        positional_encoding = positional_encoding[np.newaxis, ...]

        return positional_encoding


    if __name__ == '__main__':
        # 参数设置
        max_seq_len = maxlen  # 序列的最大长度
        embed_size = 900  # 嵌入向量的维度

        # 生成位置编码
        pos_encoding = get_positional_encoding(max_seq_len, embed_size)
        pos_encoding = np.array(pos_encoding)
    #     print(pos_encoding.shape)
    #     for i in pos_encoding:
    #         print(i)
    # exit(0)
    # 词嵌入
    cqr = np.array(cqr)
    # mask矩阵
    mask_data = tf.sequence_mask(seqlen, maxlen=maxlen)
    # print(mask_data.shape)
    # print(cqr.shape)
    # print("aldjkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # 遮蔽了排除掉无效的信息
    # 词嵌入和位置编码相加
    # for i in range(len(pos_encoding)):
    #     for j in range(len(pos_encoding[i])):
    #         print(i)
    #         print(pos_encoding[i][j])
    #         print("----------------")
    # for i in range(len(cqr)):
    #     for j in range(len(cqr[i])):
    #         print(pos_encoding[j])
    # cqr[i][j] =cqr[i][j]+pos_encoding[j]
    inputself1 = []
    inputself = []
    for i in range(len(cqr)):
        inputself1 = []
        for j in range(len(cqr[i])):
            cqr[i][j] = cqr[i][j] #+ pos_encoding[0][j]
            if mask_data[i][j] == False:
                cqr[i][j] = -100000000000000000000000000000000000
            inputself1.append(cqr[i][j])
            # print(cqr[i][j])
        inputself.append(inputself1)
    # 自注意力的输入
    inputself = np.array(inputself)
    inputself = torch.from_numpy(inputself).float()
    inputself = torch.tensor(inputself)


    class SelfAttention(nn.Module):

        def __init__(self, d_in, d_out_kq, d_out_v):
            super().__init__()
            self.d_out_kq = d_out_kq
            self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value

            attn_scores = queries @ keys.T  # unnormalized attention weights
            attn_weights = torch.softmax(
                attn_scores / self.d_out_kq ** 0.5, dim=-1
            )

            context_vec = attn_weights @ values
            return context_vec


    torch.manual_seed(123)

    # reduce d_out_v from 4 to 1, because we have 4 heads
    d_in, d_out_kq, d_out_v = 900, 900, 900
    selfout = []
    sa = SelfAttention(d_in, d_out_kq, d_out_v)
    for i in inputself:
        i = tf.squeeze(i, axis=1)
        i = np.array(i)
        i = torch.from_numpy(i).float()
        selfout.append(sa(i))
        # i=torch.tensor(i)
        # print(sa(i))
    selfout1 = []
    selfout2 = []
    import math

    for i in selfout:
        selfout1 = []
        i = i[:].detach().numpy()
        for j in i:
            # print(j)
            j = sum(j)
            selfout1.append(j)
            # print(j)
            # print("-------------------------")
        selfout1 = [0 if math.isnan(x) else x for x in selfout1]
        selfout2.append(selfout1)
    selfout2 = np.array(selfout2)

    # plt.imshow(selfout2, cmap='Reds')
    #
    # # 添加颜色条
    # plt.colorbar()
    #
    # # 显示图像
    # plt.show()

    # print(selfout2.shape)
    fealen = selfout2.shape[-1]
    in_x1 = tf.keras.layers.Input(shape=(fealen,))
    # x1 = tf.keras.layers.Embedding(21, 16)(in_x1)
    # #x1=tf.keras.layers.LSTM(216, return_sequences=True)(x1)
    # x1 = tf.keras.layers.Dense(maxlen, 'relu')(x1)  #
    # print(x1)
    x1 = tf.keras.layers.Embedding(216, 64)(in_x1)  # 序列文本转化为特征
    x1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(x1)
    # print(scores[1])
    # exit(0)
    # this is a logistic regression in Keras

    # x1=model.add(Dense(250,'relu'))
    train2 = []
    for i in train_index:
        if str(rx[i].id).endswith("1"):
             train2.append(encode(rx[i].seq))
        if str(rx[i].id).endswith("0"):
            train2.append(encode(rx[i].seq))

    test2 = []
    for i in test_index:
        if str(rx[i].id).endswith("1"):
            test2.append(encode(rx[i].seq))
        if str(rx[i].id).endswith("0"):
            test2.append(encode(rx[i].seq))

    for i in train2:
        i = np.array(i)

    # train1是第一个通道，train2是第二个通道
    # 第一个通道先embedding，再biLSTM
    # 第二个通道是直接将数据集进行特征函数编码
    # print(train2)
    selector = VarianceThreshold()
    train2 = selector.fit_transform(train2)
    test2 = selector.transform(test2)
    print(train2.shape)
    # 归一化
    train2 = preprocessing.normalize(train2, norm='l1')
    test2 = preprocessing.normalize(test2, norm='l1')
    # 标准化
    train22 = []
    for i in train2:
        mean = np.mean(i, axis=0)
        std = np.std(i, axis=0)
        i = (i - mean) / std
        train22.append(i)
    train22 = np.array(train22)
    train2 = train22
    train2 = np.array(train2)
    print(train2.shape)
    # print("--------------------------------")
    # train2=preprocessing.normalize(train2,norm='l1')
    # test2=preprocessing.normalize(test2,norm='l1')
    train2=np.array(train2)
    test2=np.array(test2)
    train21 = train2
    train21 = np.array(train21)
    test22=[]
    for i in test2:
        mean=np.mean(i,axis=0)
        std=np.std(i,axis=0)
        i=(i-mean)/std
        test22.append(i)
    test22=np.array(test22)
    test2=test22
    test21 = test2
    test21 = np.array(test21)
    train1 = np.array(train1)
    train2 = np.array(train2, ndmin=3)
    train2wei = int(train2.shape[-1])
    print(train2wei)
    train_label = np.array(train_label)
    test1 = np.array(test1)
    test2 = np.array(test2)
    x = torch.from_numpy(train2)
    #print(x.shape)
    num_heads = 0
    head_dim = 0
    a=x.shape[-1]
    for i in range(a):
        if i!=0:
            if a%i==0:
                for j in range(i):
                    if j!=0:
                        if i%j==0:
                            num_heads=j
                            head_dim=i/j
    print(num_heads,head_dim)
    if num_heads==0:
        num_heads=1
        head_dim=1
    # exit(0)
    # num_heads = 2
    # head_dim = 2
    b=a/num_heads/head_dim
    # feature_dim 必须是 num_heads * head_dim 的整数倍
    assert x.size(-1) == num_heads * head_dim * b
    x = x.float()
    # 定义线性层用于将 x 转换为 Q, K, V 向量
    # linear_q = torch.nn.Linear(856, 856)
    # linear_k = torch.nn.Linear(856, 856)
    # linear_v = torch.nn.Linear(856, 856)

    linear_q = torch.nn.Linear(train2wei, train2wei)
    linear_k = torch.nn.Linear(train2wei, train2wei)
    linear_v = torch.nn.Linear(train2wei, train2wei)
    # 通过线性层计算 Q, K, V
    Q = linear_q(x)  # 形状 (batch_size, seq_len, feature_dim)
    K = linear_k(x)  # 形状 (batch_size, seq_len, feature_dim)
    V = linear_v(x)  # 形状 (batch_size, seq_len, feature_dim)


    # 将 Q, K, V 分割成 num_heads 个头
    def split_heads(tensor, num_heads):
        batch_size, seq_len, feature_dim = tensor.size()
        head_dim = feature_dim // num_heads
        output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return output  # 形状 (batch_size, num_heads, seq_len, feature_dim)


    Q = split_heads(Q, num_heads)  # 形状 (batch_size, num_heads, seq_len, head_dim)
    K = split_heads(K, num_heads)  # 形状 (batch_size, num_heads, seq_len, head_dim)
    V = split_heads(V, num_heads)  # 形状 (batch_size, num_heads, seq_len, head_dim)
    # 计算 Q 和 K 的点积，作为相似度分数 , 也就是自注意力原始权重
    raw_weights = torch.matmul(Q, K.transpose(-2, -1))  # 形状 (batch_size, num_heads, seq_len, seq_len)
    # 对自注意力原始权重进行缩放
    scale_factor = K.size(-1) ** 0.5
    scaled_weights = raw_weights / scale_factor  # 形状 (batch_size, num_heads, seq_len, seq_len)
    # 对缩放后的权重进行 softmax 归一化，得到注意力权重
    attn_weights = F.softmax(scaled_weights, dim=-1)  # 形状 (batch_size, num_heads, seq_len, seq_len)
    # 将注意力权重应用于 V 向量，计算加权和，得到加权信息
    attn_outputs = torch.matmul(attn_weights, V)  # 形状 (batch_size, num_heads, seq_len, head_dim)


    # 将所有头的结果拼接起来
    def combine_heads(tensor, num_heads):
        batch_size, num_heads, seq_len, head_dim = tensor.size()
        feature_dim = num_heads * head_dim
        output = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
        return output  # 形状 : (batch_size, seq_len, feature_dim)


    attn_outputs = combine_heads(attn_outputs, num_heads)  # 形状 (batch_size, seq_len, feature_dim)
    # 对拼接后的结果进行线性变换
    linear_out = torch.nn.Linear(train2wei, train2wei)
    attn_outputs = linear_out(attn_outputs)  # 形状 (batch_size, seq_len, feature_dim)
    print(attn_outputs.shape)
    attn_outputs = attn_outputs[:].detach().numpy()
    attn_outputs = np.array(attn_outputs)
    attn_outputs = np.squeeze(attn_outputs)
    attn_outputs=preprocessing.normalize(attn_outputs,norm='l1')
    # zt=[]
    # for i in range(len(attn_outputs)):
    #     zl=[]
    #     zl=np.append(selfout2[i],attn_outputs[i])
    #     zt.append(zl)
    # zt=np.array(zt)
    # print(zt.shape)
    # model = TSNE(n_components=2,init='pca',random_state=0)
    # tsne = model.fit_transform(zt)
    # xmin, xmax = tsne.min(0), tsne.max(0)
    # xnor = (tsne - xmin) / (xmax - xmin)
    # s1, s2 = xnor[:291, :], xnor[291:, :]
    # print(s1)
    # print("-------------------------")
    # print(s2)
    # fig = plt.figure()
    # t1 = plt.scatter(s1[:, 0], s1[:, 1], marker='o', c='r', s=5)
    # t2 = plt.scatter(s2[:, 0], s2[:, 1], marker='o', c='g', s=5)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend((t1, t2), ('ACP', 'nonACP'))
    # plt.show()
    # exit(0)
    # 绘制热图
    # plt.imshow(attn_outputs, cmap='Reds')
    #
    # 添加颜色条
    # plt.colorbar()
    #
    # 显示图像
    # plt.show()
    # print(attn_outputs.shape)
    fealen = attn_outputs.shape[-1]
    #fealen=train21.shape[-1]
    # attn_outputs=np.reshape(attn_outputs,(-1,856))
    #train2 = np.array(attn_outputs)
    # print(train2)
    # exit(0)
    # 模型
    # 通道1是embedding biLSTM
    # 通道2是特征编码
    # 通道1 2进行concatenate
    # 进入全连接层
    # fealen = train2.shape[-1]
    # in_x1 = tf.keras.layers.Input(shape=(maxlen,))
    in_x2 = tf.keras.layers.Input(shape=(fealen,))  # 各个特征编码维度加起来的结果
    # print(in_x2)

    # exit(0)
    # x1=Conv1D(400, 1, padding='valid', activation='relu', strides=1)(in_x1)
    # x1 = GlobalMaxPooling1D()(x1)
    # x1 = tf.keras.layers.Dense(200, 'relu')(x1)
    # x1 = tf.keras.layers.Dropout(0.2)(x1)
    # x1 = tf.keras.layers.Embedding(21, 64)(in_x1)  # 序列文本转化为特征
    # x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x1)

    # x1=
    # x1=GlobalMaxPooling1D()(x1)
    # x1=tf.keras.layers.Dense(200,'relu')(x1)
    # x1=tf.keras.layers.Dropout(0.2)(x1)
    # x1=np.expand_dims(x1,axis=-1)
    # x1 = Lambda(lambda x1: K.expand_dims(l1, axis=-1))(x1)
    # x1 = tf.keras.layers.LSTM(64)(x1)
    # x1=tf.keras.layers.Dense(1,'relu')(x1)
    x2 = tf.keras.layers.Embedding(64, 64)(in_x2)
    x2 = tf.keras.layers.Dense(64, 'relu')(in_x2)  #
    #x = tf.keras.layers.Concatenate(axis=-1)([x1,x2])
    x = tf.keras.layers.Concatenate(axis=-1)([x1,x2])
    #x = tf.keras.layers.Concatenate(axis=-1)([x2])
    x=tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(train2wei, 'relu')(x)
    x = tf.keras.layers.Dense(216, 'relu')(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    out_x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    # print(x1)
    # x1=GCN
    model=Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(216,return_sequences=True)))
    model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(), return_sequences=True)))
    # model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, 'sigmoid'))
    model = tf.keras.Model(inputs=[in_x1,in_x2], outputs=[out_x])
    #model = tf.keras.Model(inputs=[in_x2], outputs=[out_x])
    #model = tf.keras.Model(inputs=[in_x2], outputs=[out_x])#in_x1,
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.fit([train1,train21], train_label,
              batch_size=batchsize,
              epochs=epochs
              )

    #model.fit([train21], train_label,

    # 预测
    pred_res = model.predict([test1,test2])
    #pred_res = model.predict([test2])
    # zt=[]
    # for x in range(len(pred_res)):
    #     zt1=[]
    #     zt1.append(test_label[x])
    #     zt1.append(float(pred_res[x]))
    #     zt.append(zt1)
    # y=[]
    # x=[]
    # for i in range(len(zt)):
    #     for j in range(len(zt[i])):
    #         if zt[i][0]==0:
    #             y.append(zt[i][1])
    #         else:
    #             x.append(zt[i][1])
    # y=np.array(y)
    # x=np.array(x)
    # mean=np.mean(x,axis=0)
    # mean1=np.mean(y,axis=0)
    # std=np.std(x,axis=0)
    # std1=np.std(y,axis=0)
    # ny=(y-mean1)/std1
    # nd=(x-mean)/std
    # a=0
    # b=0
    # for i in range(len(zt)):
    #     for j in range(len(zt[i])):
    #         if zt[i][0]==0:
    #             zt[i][1]=ny[a]
    #             a=a+1
    #         else:
    #             zt[i][1]=nd[b]
    #             b=b+1
    # #exit(0)
    # zt=np.array(zt)
    # for i in range(150):
    #     print(pred_res[i])
    #     print(train_label[i])
    #t = 0.9
    pred_label = []
    #print(pred_res)

        # print(test_label[x])
        # #print(pred_label[x])
        # print(pred_res[x])
        # print("=========================")
    for x in pred_res:
        if(x>0.9

        ):
            pred_label.append(1)
        else:
            pred_label.append(0)
    acc = metrics.accuracy_score(y_pred=pred_label, y_true=test_label)
    mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=test_label)
    specificity = calculate_specificity(len(test_label), y_pred=pred_label, y_true=test_label)
    f1 = metrics.f1_score(y_pred=pred_label, y_true=test_label)
    recall = metrics.recall_score(y_pred=pred_label, y_true=test_label)
    auc = metrics.roc_auc_score(y_true=test_label, y_score=pred_res)
    precision = metrics.precision_score(y_pred=pred_label, y_true=test_label)

    performance.append([acc, mcc, precision, recall, specificity, f1, auc])
    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred_res, pos_label=1)
    fp.append(fpr)
    tp.append(tpr)
    roc_auc.append(auc)
    print("指标：")
    print(performance)
    print("test_label：")
    print(test_label)
    print("testdate")
    print(test_data1)
    print("pred_res：")
    print(pred_res)
    print("pred_label：")
    print(pred_label)

plt.rc('font', family='Times New Roman')
plt.figure(dpi=600)
plt.plot(fp[0], tp[0], 'salmon', label='First AUC = %0.2f' % roc_auc[0])
plt.plot(fp[1], tp[1], 'paleturquoise', label='Second AUC = %0.2f' % roc_auc[1])
plt.plot(fp[2], tp[2], 'violet', label='Third AUC = %0.2f' % roc_auc[2])
plt.plot(fp[3], tp[3], 'palegreen', label='Forth AUC = %0.2f' % roc_auc[3])
plt.plot(fp[4], tp[4], 'royalblue', label='Fifth AUC = %0.2f' % roc_auc[4])

plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'size': 15})
plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'size': 15})
plt.title('ACP740 5Fold ROC', fontdict={'family': 'Times New Roman', 'size': 20})
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

# 综合评价即五次结果的均值
mean = np.mean(np.array(performance), axis=0)
a = 1
for i in performance:
    print('第', a, '次 acc  mcc  pre  se  sp  f1  auc')
    print(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
    a += 1

print("平均指标:acc  mcc  pre  se  sp  f1  auc")
print(mean)

# print(embedded_sentence)
# print(embedded_sentence.shape)

