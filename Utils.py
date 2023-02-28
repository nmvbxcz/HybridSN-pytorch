import numpy as np
from sklearn import metrics, preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
import torch
import cv2
from operator import truediv


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def set_figsize(figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices, path):
    pred_test = []
    for X, y in all_iter:
        # X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        pred_test.extend(net(X).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 300,
                       path + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '_gt.png')
    print('------Get classification maps successful-------')


def chooseRSdata(dataSetName, train_ratio, normType=1, trainId=0):
    # 输入：
    #     dataSetName:高光谱数据集名称。
    #     train_ratio:训练样本个数，大于一时按个数取，小于1时按百分比取，每一类样本最多取到该类样本总数的一半。
    #     normType:标准化方式(0:减均值;1：z-score归一化;2： 最大最小值归一化;3:除以最大值归一化)
    #     radius:均值滤波半径
    #     trainId：训练集种子序号（0-9，对应randp文件中的随机选择训练集的种子序号）

    # 数据集下载地址（课题组内请从小组QQ群文件下载）：
    # Indian_pines: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines
    # PaviaU: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene
    # Salinas:https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene
    # Houston2013:https://hyperspectral.ee.uh.edu/?page_id=459
    # Houston2018:http://hyperspectral.ee.uh.edu/?page_id=1075
    # Hyrank(Loukia):https://aistudio.baidu.com/aistudio/datasetdetail/80840
    # HanChuan, HongHu, LongKou:http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
    # xuzhou:https://ieee-dataport.org/documents/xuzhou-hyspex-dataset

    # 使用示例：
    # for i in range(10):  # 训练10次
    #      return XTrain,XTest,YTrain,YTest = chooseRSdata(dataSetName='Indianpines', train_ratio=10, normType=1,radius=0, trainId=i)

    data_path = os.path.join(os.getcwd(), 'dataset/')
    randp_path = os.path.join(os.getcwd(), 'dataset/randp')
    if dataSetName == 'IN':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        randp = sio.loadmat(os.path.join(randp_path, 'Indian_pines_randp.mat'))['randp']
    elif dataSetName == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        randp = sio.loadmat(os.path.join(randp_path, 'PaviaU_randp.mat'))['randp']
    elif dataSetName == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        randp = sio.loadmat(os.path.join(randp_path, 'Salinas_randp.mat'))['randp']
    elif dataSetName == 'Houston2013'.upper():
        data = sio.loadmat(os.path.join(data_path, 'Houston2013.mat'))['CASI']
        labels = sio.loadmat(os.path.join(data_path, 'Houston2013_gt.mat'))['gnd_flag']
        randp = sio.loadmat(os.path.join(randp_path, 'Houston2013_randp.mat'))['randp']
    elif dataSetName == 'Houston2018'.upper():
        data = sio.loadmat(os.path.join(data_path, 'HOU2018_correct'))['img']
        labels = sio.loadmat(os.path.join(data_path, 'HOU2018_GT.mat'))['GT']
        randp = sio.loadmat(os.path.join(randp_path, 'Houston2018_randp.mat'))['randp']
    elif dataSetName == 'Loukia'.upper():
        data = io.imread(os.path.join(data_path, 'Loukia.tif'))
        labels = io.imread(os.path.join(data_path, 'Loukia_GT.tif'))
        randp = sio.loadmat(os.path.join(randp_path, 'Loukia_randp.mat'))['randp']
    elif dataSetName == 'HanChuan'.upper():
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
        randp = sio.loadmat(os.path.join(randp_path, 'WHU_Hi_HanChuan_randp.mat'))['randp']
    elif dataSetName == 'HongHu'.upper():
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
        randp = sio.loadmat(os.path.join(randp_path, 'WHU_Hi_HongHu_randp.mat'))['randp']
    elif dataSetName == 'LongKou'.upper():
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
        randp = sio.loadmat(os.path.join(randp_path, 'WHU_Hi_LongKou_randp.mat'))['randp']
    elif dataSetName == 'Xuzhou'.upper():
        data = sio.loadmat(os.path.join(data_path, 'Xuzhou.mat'))['all_x']
        labels = sio.loadmat(os.path.join(data_path, 'Xuzhou.mat'))['all_y']
        randp = sio.loadmat(os.path.join(randp_path, 'Xuzhou_randp.mat'))['randp']
        data = data.reshape(260, 500, 436)
        data = data.transpose((1, 0, 2))
        labels = labels.reshape(260, 500).T
    else:
        print("Unknown data set requested.")
        sys.exit()

    data = np.double(data)

    h, w, c = data.shape
    data = data.transpose((1, 0, 2))  # 适配matlab的行优先和列优先
    data = np.reshape(data, [h * w, c])

    # 标准化方式
    if normType == 0:  # 减均值
        xx = np.zeros([h * w, c])
        bias = np.mean(data, axis=0)
        for i in range(c):
            xx[:, i] = data[:, i] - bias[i]
        data = xx
    elif normType == 1:
        Scaler = preprocessing.StandardScaler()  # z-score归一化
        data = Scaler.fit_transform(data)
    elif normType == 2:
        min_max_scaler = preprocessing.MinMaxScaler()  # 最大最小值归一化
        data = min_max_scaler.fit_transform(data)
    elif normType == 3:
        Normalize = np.max(data, axis=0)  # 除以最大值归一化
        xx = data / np.tile(Normalize, (data.shape[0], 1))
        data = xx
    else:
        print("Wrong parameters for normalization.")
        sys.exit()

    labels = labels.T  # 适配matlab的行优先和列优先
    labels = np.reshape(labels, [labels.shape[0] * labels.shape[1]])
    all_class = randp[:, trainId][0]

    for label in range(np.max(labels)):
        # class_num = all_class[label][0]
        index = all_class[0, label] - 1  # 随机序列,-1适配matlab下标从1开始
        index = index.reshape(index.shape[1])
        indexList = np.where(labels == label + 1)[0]  # 顺序序列,去除0标签
        ci = len(indexList)
        datai = data[indexList]
        if train_ratio != 1:
            cTrain = max(int((1 - train_ratio) * len(indexList)), 3)
        else:
            cTrain = 0
        # if train_ratio > 1:
        #     cTrain = round(train_ratio)
        # elif train_ratio < 1:
        #     cTrain = round(ci * train_ratio)
        # if train_ratio > round(ci / 2):
        #     cTrain = round(ci / 2)
        cTest = ci - cTrain
        rs_test_index = np.squeeze(index[0:cTest].astype('int64'))
        rs_train_index = np.squeeze(index[cTest:cTest + cTrain].astype('int64'))
        if label == 0:
            train_ind = indexList[rs_train_index]
            test_ind = indexList[rs_test_index]
        else:
            train_ind = np.append(train_ind, indexList[rs_train_index], axis=0)
            test_ind = np.append(test_ind, indexList[rs_test_index], axis=0)
        train_gt = np.zeros_like(labels)
        train_gt[train_ind] = labels[train_ind]
        test_gt = np.zeros_like(labels)
        test_gt[test_ind] = labels[test_ind]
    return train_ind, test_ind, train_gt.reshape(h,w).T, test_gt.reshape(h,w).T