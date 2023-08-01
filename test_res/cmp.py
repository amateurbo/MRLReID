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
    f = open('mta_reid_dif_dict.txt', 'r')
    line = f.read()

    dic = eval(line)
    # sorted(dic)
    # 对字典进行降序排序
    # desc_dict = {k: v for k, v in sorted(dic.items(), key=lambda x: x[0] / x[1][2], reverse=True)}
    desc_dict = {k: v for k, v in sorted(dic.items(), key=lambda x: x[0], reverse=True)}
    value_list = list(desc_dict.values())
    # print("frist 100")
    # print(value_list[:100])
    # print('\n\n\n')
    #
    # l = int(len(value_list) / 2)
    # print("mid 100")
    # print(value_list[l - 50:l + 50])
    # print('\n\n\n')
    #
    # print("last 100")
    # print(value_list[-100:])
    # print('\n\n\n')
    for k, v in desc_dict.items() :
        # if k < 5.0 and k > -5.0:
        #     print(k, v)
        if k > 30 or k < -30:
            print(k, v)

    # print(desc_dict)
    # prelist = dic['pre']
    # tarlist = dic['tar']
    # cmp_PLCC(prelist, tarlist)
    # PLCC = scipy.stats.pearsonr(prelist, tarlist)
    # SROCC = scipy.stats.spearmanr(prelist, tarlist)
    # KROCC = scipy.stats.kendalltau(prelist, tarlist)
    # RMSE(prelist, tarlist)

    # print(PLCC)
    # print(SROCC)
    # print(KROCC)
# print(dic)