'''
    Instance gravity oversampling method for software defect prediction
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KernelDensity, LocalOutlierFactor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
import utils.smote_variant_2 as smote_var
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
import time
import utils.Evaluation_indexs as EI
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(X):
    """
    预处理数据，处理无穷大值和数值过大的情况
    """
    X = np.nan_to_num(X, nan=0.0)  # 将NaN值替换为0
    X[np.isinf(X)] = np.finfo(np.float64).max  # 将无穷大值替换为float64的最大值
    X = np.clip(X, -np.finfo(np.float64).max, np.finfo(np.float64).max)  # 限制数据在float64范围内
    return X



def calculate_gravitation(data_particle1, data_particle2):
    '''
    计算两个数据粒子之间的引力
    :param data_particle1:
    :param data_particle2:
    :return:
    '''
    data_particle1, data_particle2 = np.array(data_particle1), np.array(data_particle2)
    # print(data_particle1)
    # 计算每个样本的质量，规定所有特征值之和为样本质量
    m1, m2 = np.sum(data_particle1), np.sum(data_particle2)
    # print(m1, m2)
    r = np.linalg.norm(data_particle1 - data_particle2)  # 计算两个中心点之间的欧氏距离
    if r == 0:  # 如果距离为0，则返回0以避免除零错误
        return 0
    return (m1 * m2) / (r ** 2)  # 计算并返回引力值，根据万有引力公式



def calculate_total_gravitation(single_particle, data_particles):
    '''
    计算单个样本与数据集中所有样本的总引力
    :param single_particle:
    :param data_particles:
    :return:
    '''
    total_gravitation = 0
    for particle in data_particles:
        total_gravitation += calculate_gravitation(single_particle, particle)
    return total_gravitation



def random_selection(X_train_min, syn_min):
    '''
    从少数类样本集中随机选取一个样本，并从合成少数类样本集中随机选取一个样本
    :param X_train_min:
    :param syn_min:
    :return:
    '''
    random_index_min = np.random.randint(0, len(X_train_min))  # 生成随机索引
    random_index_syn = np.random.randint(0, len(syn_min))  # 生成随机索引
    random_sample_min = X_train_min[random_index_min]  # 使用索引从少数类样本集中选择样本
    random_sample_syn = syn_min[random_index_syn]  # 使用索引从合成少数类样本集中选择样本
    return random_sample_min, random_sample_syn


def MOS(X_train, y_train):
    '''
    特征模型生成合成样本
    :param X_train:
    :param y_train:
    :return:
    '''
    X_train_minor, X_train_major = X_train[y_train == 1], X_train[y_train == 0]

    iter = 5  # 迭代次数
    syn_minor = model_based_synthetic_sampling_correct(X_train_minor, iter)
    syn_label = np.ones(syn_minor.shape[0])

    return syn_minor, syn_label


def model_based_synthetic_sampling_correct(X_minority, iterations):
    """
        正确实现特征模型算法，考虑特征之间的关系，为每个特征生成合成数据。
    """
    # print(X_minority, X_minority.shape)
    # print('*' * 50)
    n_samples, n_features = X_minority.shape
    synthetic_samples = np.zeros_like(X_minority)

    # 逐个特征进行处理
    for feature_idx in range(n_features):
        # 使用除了当前特征外的其他所有特征作为输入，当前特征作为输出
        X_train = np.delete(X_minority, obj=feature_idx, axis=1).astype('float64')
        y_train = X_minority[:, feature_idx]

        model = SVR() # cart yes
        model.fit(X_train, y_train)

        # 生成临时样本，这里直接复用其他特征的值，只预测当前特征
        temp_synthetic_features = np.random.choice(X_minority[:, feature_idx], size=n_samples)
        synthetic_samples[:, feature_idx] = temp_synthetic_features

    # 迭代预测每个特征
    for _ in range(iterations):
        # 使用训练好的模型预测每个特征的值
        for feature_idx in range(n_features):
            X_pred = np.delete(synthetic_samples, obj=feature_idx, axis=1) # 准备预测输入：排除当前预测特征
            X_pred = preprocess_data(X_pred) # 预处理预测数据，防止出现无穷大值
            synthetic_samples[:, feature_idx] = model.predict(X_pred) # 预测并更新合成样本的当前特征值

    # 处理模型合成样本
    synthetic_samples = preprocess_data(synthetic_samples)

    return synthetic_samples


def MOSIG(X_train, y_train):

    X_train_minor, X_train_major = X_train[y_train == 1], X_train[y_train == 0] # 原始数据集中的少数类

    syn_minor, syn_label = MOS(X_train, y_train) # 模型合成的少数类

    # 计算每个原始少数类样本的引力
    original_gravities = np.array([calculate_gravitation(sample, syn_minor) for sample in X_train_minor])
    # 计算每个模型合成少数类样本的引力
    synthetic_gravities = np.array([calculate_gravitation(sample, X_train_minor) for sample in syn_minor])

    syn_sample = []
    while len(X_train_minor) + len(syn_sample) < len(X_train_major):
        # 随机从原始数据集中选择一个少数类样本
        random_index = np.random.randint(0, len(X_train_minor))
        random_sample = X_train_minor[random_index]
        random_gravity = calculate_gravitation(random_sample, syn_minor)

        # 找到模型合成少数类中引力与随机选择少数类样本引力差值最小的样本
        min_diff_index = np.argmin(np.abs(synthetic_gravities - random_gravity))
        closest_synthetic_sample = syn_minor[min_diff_index]
        syn_random_gravity = calculate_gravitation(closest_synthetic_sample, X_train_minor)

        # 计算权重
        X_weight, syn_weight = random_gravity / (random_gravity + syn_random_gravity), syn_random_gravity / (
                    random_gravity + syn_random_gravity)

        # 合成少数类样本
        new_syn_sample = np.multiply(X_weight, random_sample) + np.multiply(syn_weight, closest_synthetic_sample)
        syn_sample.append(new_syn_sample)

        # 生成第二个样本
        # 随机从模型数据集中选择一个少数类样本
        random_index = np.random.randint(0, len(syn_minor))
        syn_random_sample = syn_minor[random_index]
        random_gravity = calculate_gravitation(syn_random_sample, X_train_minor)

        # 找到原始少数类中引力与模型合成少数类样本引力差值最小的样本
        min_diff_index = np.argmin(np.abs(original_gravities - random_gravity))
        closest_synthetic_sample = X_train_minor[min_diff_index]
        syn_random_gravity = calculate_gravitation(closest_synthetic_sample, syn_minor)

        # 计算权重
        X_weight, syn_weight = random_gravity / (random_gravity + syn_random_gravity), syn_random_gravity / (
                random_gravity + syn_random_gravity)

        # 合成少数类样本
        new_syn_sample = np.multiply(X_weight, random_sample) + np.multiply(syn_weight, closest_synthetic_sample)
        syn_sample.append(new_syn_sample)

    syn_sample = np.array(syn_sample)
    # 处理引力合成样本
    syn_sample = preprocess_data(syn_sample)
    syn_sample_label = np.ones(len(syn_sample))

    # 使用LOF删除合成少数类噪声
    lof = LocalOutlierFactor()
    y_lof = lof.fit_predict(syn_sample)
    # 保留LOF识别为非噪声的样本
    mask = y_lof != -1
    # LOF处理后的少数类样本
    syn_minor_lof, syn_label_lof = syn_sample[mask], syn_sample_label[mask]
    # print('LOF 删除少数类样本个数', (len(syn_sample) - len(syn_minor_lof)))

    X_train = np.concatenate((X_train, syn_minor_lof), axis=0)
    y_train = np.concatenate((y_train, syn_label_lof), axis=0)

    return X_train, y_train


if __name__ == '__main__':

    start = time.time()
    data_frame = np.array(pd.read_csv(r'D:\A论文实验\data\NASA\PC1.csv')) # 路径
    print('样本个数: ', data_frame.shape[0], '特征个数: ', data_frame.shape[1] - 1)
    data = data_frame[:, :-1].astype('float64')
    target = data_frame[:, -1]
    print('样本不平衡比', (sum(target == 0) / sum(target == 1)))

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = preprocess_data(data)

    F_measure, TPR, FNR, G_mean, MCC, AUC = [], [], [], [], [], []

    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # shuffle=True 随机划分
    for train_index, test_index in kfold.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        X_train_LOF, y_train_LOF = MOSIG(X_train, y_train)

        model = DecisionTreeClassifier() # CART
        model.fit(X_train_LOF, y_train_LOF)
        pred = model.predict(X_test)
        f, tpr, fnr, gmean, mcc, auc = EI.evaluation_indexs(y_test, pred)
        F_measure.append(f)
        TPR.append(tpr)
        FNR.append(fnr)
        G_mean.append(gmean)
        MCC.append(mcc)
        AUC.append(auc)

    print('*' * 25, 'CART', '*' * 25)
    print('MOSIG F-measure: %.4f' % np.mean(F_measure_LOF))
    print('MOSIG TPR: %.4f' % np.mean(TPR_LOF))
    print('MOSIG FNR: %.4f' % np.mean(FNR_LOF))
    print('MOSIG G-mean: %.4f' % np.mean(G_mean_LOF))
    print('MOSIG MCC: %.4f' % np.mean(MCC_LOF))
    print('MOSIG AUC: %.4f' % np.mean(AUC_LOF))

    end = time.time()
    print('运行时间%.4fs'%(end - start))
