import numpy as np
import Block_diagonal
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

#Get the set P which are classified as false negative
def getSetP(y, y_hat):
    idx = np.where(y_hat == -1)
    P = []
    for index in idx[0]:
        if y[index] == 1:
            P.append(index)
    return P

#D is the distance between each sample in P and hyperplane
def getDistanceD(P, x, w, b):
    D = {}
    for index in P:
        dis = float(abs(np.dot(w.T, x[index].T) + b) / np.dot(w.T,w.T))
        D[index] = dis
    return D

#A is the sample which has the maxmum in D
def getMaxmumA(D):
    temp = sorted(D.items(),key=lambda item:item[1])
    return temp[-1]

#Select a sample B which is classified as true positive and is nearest to A
def getNearestB(A, x, y, y_hat):
    id_A = A[0]
    id_B = 0
    id_tmp = np.where(y_hat == 1)
    dis_B = 99999
    for index in id_tmp[0]:
        if y[index] == 1:
            #cal the distance between A and this point
            dis = np.linalg.norm(x[id_A] - x[index])
            if dis < dis_B:
                dis_B = dis
                id_B = index
            #print (dis)
    return (id_B, dis_B)
    
#Get the similarity Matrix
def getSimilarity(x, y, k=2):
    data = (x - x.mean(0)) / x.std(0)
    M = Block_diagonal.block_diagonal(data, label = y, K = k)
    #M.fit()
    # Show the resulted similarity matrix.
    #M.show_matrix()
    #plt.show()
    return M.returnsW()

#Select a sample C which is classified as true positive and is nearest to B
def getNearestC(B, x, y, y_hat):
    id_B = B[0]
    id_C = 0
    id_tmp = np.where(y_hat == 1)
    dis_C = 99999
    for index in id_tmp[0]:
        if y[index] == 1 and index != id_B:
            #cal the distance between B and this point
            dis = np.linalg.norm(x[id_B] - x[index])
            if dis < dis_C:
                dis_C = dis
                id_C = index
    return (id_C,dis_C)

#Get PAB and PBC
def getPabAndPbc(KAB, KBC, matrix, x, y, k=2):
    M = Block_diagonal.block_diagonal(matrix, label = y, K = k,
                                      similarity_matrix_given = True)
    M.fit()
    Pab = M.s_old(KAB)
    Pbc = M.s_new(KBC)
    return (Pab, Pbc)

#cal the lower bound of C+
def getLowerBoundOfCplus(KAB, PAB, PBC, k=1):
    return float(PAB / (4 * KAB * (PAB - k * PBC) - PAB ))

#undersampling data
def undersample(data):
    data_pos, data_neg = [], []
    data_more, data_less = [], []
    data_new = []
    for item in data:
        if int(item[-1]) == 1:
            data_pos.append(item)
        elif int(item[-1]) == -1:
            data_neg.append(item)
    data_pos_len = len(data_pos)
    data_neg_len = len(data_neg)
    if data_pos_len < data_neg_len:
        data_more = data_neg
        data_less = data_pos
    else:
        data_more = data_pos
        data_less = data_neg
    IR = float(len(data_more) / len(data_less))
    np.random.shuffle(data_more)
    data_more = data_more[:len(data_less)]
    data_new = data_more + data_less
    return np.array(data_new).astype(float), IR
        
