import os
import math
import scipy
import scipy.stats


def cmp_PLCC(prelist, tarlist):
    pre_mean = sum(prelist) / len(prelist)
    tar_mean = sum(tarlist) / len(tarlist)
    suma = 0
    sumb1 = 0
    sumb2 = 0
    for pre, tar in zip(prelist, tarlist):
        suma += (tar - tar_mean) * (pre - pre_mean)
        sumb1 += (pre - pre_mean) * (pre - pre_mean)
        sumb2 += (tar - tar_mean) * (tar - tar_mean)
    print("PLCC:{}".format(suma / math.sqrt(sumb1 * sumb2)))

def cmp_SROCC(prelist, tarlist):
    l = len(prelist)


def cmp_RMSE(prelist, tarlist):
    l = len(prelist)
def RMSE(prelist, tarlist):
    l = len(prelist)
    sum = 0
    for pre, tar in zip(prelist, tarlist):
        sum += (pre - tar) * (pre - tar)

    print("RMSE:{}".format(math.sqrt(sum / l)))


if __name__ == '__main__':
    # f = open('./CAVIAR_bestRank1/9.txt', 'r')
    f = open('mta_reid_dic.txt', 'r')
    line = f.read()

    dic = eval(line)
    prelist = dic['pre']
    tarlist = dic['tar']
    cmp_PLCC(prelist, tarlist)
    PLCC = scipy.stats.pearsonr(prelist, tarlist)
    SROCC = scipy.stats.spearmanr(prelist, tarlist)

    print(PLCC)
    print(SROCC)
# print(dic)
