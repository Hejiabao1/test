import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import List, Tuple
from sklearn.model_selection import KFold

class MyDataset(Dataset):
    def __init__(self, data, labels):
        """
        data:
            np.ndarray (n_samples, n_channels, samples_size)
        labels:
            np.ndarray (n_samples)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        """
        idx:
        return: tensor类型 (sample, label)
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def dataset_concatenate(datasets: List[MyDataset]) -> MyDataset:

    data = []
    labels = []
    for a_set in datasets:
        data.append(a_set.data)
        labels.append(a_set.labels)
    data = torch.concatenate(data, dim=0)
    labels = torch.concatenate(labels, dim=0)
    return MyDataset(data, labels)


def dataset_split(dataset: MyDataset, val_rate: float) -> Tuple[MyDataset, MyDataset]:#该函数接受一个MyDataset对象的列表作为输入参数，并返回一个新的MyDataset对象
    """
    对数据集进行分割，例如将一个数据集分割为训练集与验证集
    :param dataset:
    :param val_rate:
    :return:
    """
    train_data, val_data, train_labels, val_labels = train_test_split(dataset.data, dataset.labels,
                                                                      test_size=val_rate,
                                                                      random_state=0,
                                                                      shuffle=True,
                                                                      stratify=dataset.labels)
    train_set = MyDataset(train_data, train_labels)
    val_set = MyDataset(val_data, val_labels)
    return train_set, val_set


class BME(object):
    NUM_SUBJECTS = 30
    NUM_TEST_SUBJECTS = 15
    EEG_ECG_FS = 250
    TRAIN_DATA_DIR = r'D:\Pytharm\py_Project\Depression_Detect_new\Depression_Detect\EEG_Depression\data\traindata'
    TEST_DATA_DIR = r'D:\Pytharm\py_Project\Depression_Detect_new\Depression_Detect\EEG_Depression\data\testdata'
    SAVE_DIR = './allsamples'
    TEST_SAMPLES_SAVE_DIR = './allsamples/test'
    REMARK = 'ID1-ID22为健康对照组（label为1），ID23-ID30为抑郁症患者（label为2）'

    def __init__(self, time_len, time_step):
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        if not os.path.exists(self.TEST_SAMPLES_SAVE_DIR):
            os.makedirs(self.TEST_SAMPLES_SAVE_DIR)

        self.window_size = int(self.EEG_ECG_FS * time_len)
        self.window_step = int(self.EEG_ECG_FS * time_step)

    def processing(self):

        # ******* 处理训练数据 *********
        train_data_iter = self.data_generator(self.TRAIN_DATA_DIR)
        for idx, data in enumerate(train_data_iter):
            samples = self.data_split(data)
            labels = self.generate_labels(idx=idx, n_samples=samples.shape[0])
            with open(os.path.join(self.SAVE_DIR, f'{idx:02d}.pkl'), 'wb') as f:
                pickle.dump([samples, labels], f)

        # ******* 处理测试数据 *********
        test_data_iter = self.data_generator(self.TEST_DATA_DIR)
        for idx, data in enumerate(test_data_iter):
            samples = self.data_split(data)
            with open(os.path.join(self.TEST_SAMPLES_SAVE_DIR, f'{idx:02d}.pkl'), 'wb') as f:
                pickle.dump(samples, f)

    def generate_labels(self, idx, n_samples):
        labels = None
        if idx in range(0, 21):
            labels = np.repeat(np.array([0]), repeats=n_samples)
        elif idx in range(21, 30):
            labels = np.repeat(np.array([1]), repeats=n_samples)
        else:
            Exception("ERROR: generate_labels")
        return labels

    def data_split(self, data):  # 将数据按照指定的窗口大小和步长进行分段切割。
        shape = data.shape  # 获取输入数据的形状。
        seq_len = shape[-1]  # 获取数据的序列长度，即数据的最后一个维度的长度

        data = np.reshape(data, [-1, shape[-1]])  # 将不同维度数的数据转化为[****, seq维度]的形式，

        num_segments = (seq_len - self.window_size) // self.window_step + 1  # 分割成的段数

        divided_data = []  # 创建一个空列表，用于存储分割后的数据段
        for i in range(0, num_segments * self.window_step, self.window_step):  # 循环遍历每个数据段的起始位置，以步长self.window_step进行切割
            segment = data[:, i: (i + self.window_size)]  # 取出当前段的数据，从第i列到第i+self.window_size列
            divided_data.append(segment)  # 将当前段的数据添加到divided_data列表中
        result = np.stack(divided_data, axis=-1)  # 将分割后的数据段堆叠在一起，沿着最后一个维度堆叠
        result = np.transpose(result, (2, 0, 1))  # 将数据的维度进行转置，将最后一个维度移动到第一个维度，以便后续处理
        return result


    def data_generator(self, root_dir):
        for root, dirs, files in os.walk(root_dir):
            for dir in dirs:
                eeg_path = os.path.join(root, dir, 'eegsignals.csv')
                ecg_path = os.path.join(root, dir, 'ecgsignals.csv')
                if os.path.exists(eeg_path) and os.path.exists(ecg_path):
                    eeg_data = np.genfromtxt(fname=eeg_path, delimiter=',')[1:].transpose(1, 0)
                    ecg_data = np.genfromtxt(fname=ecg_path, delimiter=',')[1:][np.newaxis, :]
                    yield np.concatenate([eeg_data, ecg_data], axis=0)


    # ******* 加载EEG数据和ECG数据*********

    def load(self, sub_index: List[int] = None) -> MyDataset:
        """
        加载processing后的数据
        :param sub_index:  list like [1, 2, 3, 5 ]
        :return: MyDataset类型
        """
        #sub_index = sub_index or range(0, self.NUM_SUBJECTS)
        if sub_index is None or len(sub_index) == 0:  # 检查sub_index是否为None或为空列表
            sub_index = range(0, self.NUM_SUBJECTS)
        samples = []
        labels = []
        for idx in sub_index:
            with open(os.path.join(self.SAVE_DIR, f'{idx:02d}.pkl'), 'rb') as f:
                x, y = pickle.load(f)
            samples.append(x)
            labels.append(y)
        samples = np.concatenate(samples, axis=0)
        labels = np.concatenate(labels, axis=0)
        return MyDataset(samples, labels)


    def load_test_data(self, sub_index: List[int] = None) -> torch.FloatTensor:
        sub_index = sub_index or range(0, 15)#如果没有提供sub_index参数，则默认为range(0, 15)，即加载前15个数据
        test_samples = []#初始化空列表用于存储测试样本数据
        for idx in sub_index:
            with open(os.path.join(self.TEST_SAMPLES_SAVE_DIR, f'{idx:02d}.pkl'), 'rb') as f:
                x = pickle.load(f)
            test_samples.append(x)
        return torch.FloatTensor(np.concatenate(test_samples, axis=0))#将测试样本列表中的数据沿着第0维进行拼接，然后转换为torch.FloatTensor类型并返回。


if __name__ == '__main__':

    bme = BME(time_len=3, time_step=2)
    # bme.processing()  # 如果尚未处理数据，先运行此行
    test_dataset = bme.load_test_data()

    # 由于测试数据集直接返回的是一个torch.FloatTensor，我们可以直接查看其形状
    print(f"Test Data Shape: {test_dataset.shape}")
    train_dataset = bme.load()  # 加载训练数据集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)  # 为了展示单个样本，这里设置batch_size=1

# 打印样本数据的维度信息
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Sample shape: {data.shape}")  # 输出样本数据的维度信息
        break  # 只打印第一个样本的信息