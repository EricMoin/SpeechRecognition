# **实验目标：**
# 通过本实验，你将深入了解和实践说话人识别技术，并掌握利用声音特征进行有效说话人识别的基本方法，了解不同特征和模型对识别准确率的影响。
# 实验的核心目标是使用TIMIT数据集来训练一个说话人识别系统，涵盖数据预处理、特征提取、模型训练和评估等关键步骤。
# **实验方法：**
# **1. 数据预处理和划分(可选)：**
#   - 数据集下载地址（4月17日前有效）：https://f.ws59.cn/f/du8yd2536vl
#   - 为了方便大家，我们提供了划分好的TIMIT数据集结构，当然你也可以根据需求自行划分该数据集。
#   - 为简化难度，我们排除了SA的两个方言句子，并在剩余的8个句子中选取了SX的5个句子和SI的1个句子作为训练集，SI的另外2个句子作为测试集。
#   - 该链接下载的数据集只保留了音频文件，完整数据集（包含音频对应文本、标注等信息）可参见备注链接下载。
# **2. 特征提取：**
#   - 学习并实现包括但不限于MFCC特征等特征的提取，探索声音信号的频率和时间特性。
#   - 鼓励尝试和比较其他特征提取方法，例如LPCC或声谱图特征，以理解不同特征对识别性能的影响。
# **3. 模型选择和训练：**
#   - 探索并选择适合的分类器和模型进行说话人识别，如GMM、Softmax分类器或深度学习模型。
#   - 实现模型训练流程，使用训练集数据训练模型。
# **4. 评估和分析：**
#   - 使用准确率作为主要的评价指标在测试集上评估模型性能。
#   - 对比不同特征和模型的性能，分析其对说话人识别准确率的影响。
#   - 可视化不同模型的识别结果和错误率，讨论可能的改进方法。
# **实验要求：**
#   - 1.选择并实现至少一种特征的提取，并鼓励尝试其他特征提取方法。
#   - 2.选择并实现至少一种分类器或模型进行说话人识别，并使用准确率评估指标评估其性能。
#   - 3.通过实验对比、分析和可视化，撰写详细的实验报告，包括实验目的、实验方法、结果分析和结论。
#   - 4.实验报告应以清晰、逻辑性强的形式呈现，图表和结果应清楚明了。

## 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import librosa
import os
import glob
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import pickle
import seaborn as sns

# 可以根据需要导入其他库，比如librosa用于音频处理
# 数据集基本信息如下
# 方言地区：DR1～DR8
# 性别：F/M
# 说话者ID：3个大写字母+1个阿拉伯数字
# 句子ID：句子类型（SA/SI/SX）+编号

# 上述链接下载的数据集已经
TrainDir = "Dataset/TRAIN"
TestDir = "Dataset/TEST"
## 请在这里写代码加载我们划分好的TIMIT训练集和测试集。或者原始完整版数据集。

# 1. 数据加载函数
def load_speaker_data(data_dir):
    """
    加载所有说话人的音频数据和对应的标签
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        speakers_data: 包含所有说话人音频路径的字典
        speakers_list: 所有说话人ID的列表
    """
    speakers_data = {}
    
    # 遍历所有方言区域
    for dialect_region in os.listdir(data_dir):
        dialect_path = os.path.join(data_dir, dialect_region)
        if not os.path.isdir(dialect_path):
            continue
            
        # 遍历每个说话人
        for speaker_id in os.listdir(dialect_path):
            speaker_path = os.path.join(dialect_path, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
                
            # 获取说话人的所有音频文件
            audio_files = glob.glob(os.path.join(speaker_path, "*.wav"))
            
            if speaker_id not in speakers_data:
                speakers_data[speaker_id] = []
                
            speakers_data[speaker_id].extend(audio_files)
    
    speakers_list = list(speakers_data.keys())
    print(f"共加载了 {len(speakers_list)} 个说话人的数据")
    
    return speakers_data, speakers_list

# 2. 特征提取函数
def extract_mfcc_features(audio_file, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    从音频文件中提取MFCC特征
    
    Args:
        audio_file: 音频文件路径
        n_mfcc: MFCC系数数量
        n_fft: FFT窗口大小
        hop_length: 帧移大小
        
    Returns:
        mfcc_features: MFCC特征
    """
    # 加载音频文件
    audio, sr = librosa.load(audio_file, sr=None)
    
    # 提取MFCC特征
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, 
                                         n_fft=n_fft, hop_length=hop_length)
    
    # 计算MFCC特征的均值，得到固定长度的特征向量
    mfcc_features = np.mean(mfcc_features, axis=1)
    
    return mfcc_features

# 3. 提取所有说话人的特征
def extract_features_for_all_speakers(speakers_data, feature_function):
    """
    为所有说话人提取特征
    
    Args:
        speakers_data: 包含所有说话人音频路径的字典
        feature_function: 特征提取函数
        
    Returns:
        features_dict: 包含所有说话人特征的字典
    """
    features_dict = {}
    
    for speaker_id, audio_files in tqdm(speakers_data.items(), desc="提取特征"):
        features_dict[speaker_id] = []
        
        for audio_file in audio_files:
            features = feature_function(audio_file)
            features_dict[speaker_id].append(features)
    
    return features_dict

# 4. 训练GMM模型
def train_gmm_models(features_dict, n_components=8, reg_covar=1e-2):
    """
    为每个说话人训练GMM模型
    
    Args:
        features_dict: 包含所有说话人特征的字典
        n_components: GMM组件数量
        reg_covar: 协方差矩阵正则化参数
        
    Returns:
        gmm_models: 包含所有说话人GMM模型的字典
    """
    gmm_models = {}
    
    for speaker_id, features in tqdm(features_dict.items(), desc="训练GMM模型"):
        # 将特征列表转换为二维数组
        X = np.array(features)
        
        # 根据样本数量动态调整组件数量
        # GMM组件数量不能超过样本数量
        actual_n_components = min(n_components, max(1, X.shape[0] - 1))  # 至少保留1个组件，且不超过样本数-1
        if actual_n_components < n_components:
            print(f"警告: 说话人 {speaker_id} 的样本数量为 {X.shape[0]}，调整组件数量为 {actual_n_components}。")
        
        # 检查特征是否包含NaN或无限值
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"警告: 说话人 {speaker_id} 的特征包含NaN或无限值，已替换为0")
            X = np.nan_to_num(X)
        
        # 训练GMM模型，增加正则化参数
        try:
            gmm = GaussianMixture(
                n_components=actual_n_components, 
                covariance_type='diag',
                random_state=42, 
                max_iter=200,
                reg_covar=reg_covar  # 增加协方差正则化参数
            )
            gmm.fit(X.reshape(X.shape[0], -1))
            gmm_models[speaker_id] = gmm
        except Exception as e:
            print(f"错误: 无法为说话人 {speaker_id} 训练GMM模型: {str(e)}")
            # 使用单高斯模型作为后备选项
            gmm = GaussianMixture(
                n_components=1, 
                covariance_type='full',
                random_state=42, 
                max_iter=200,
                reg_covar=reg_covar*10  # 更强的正则化
            )
            gmm.fit(X.reshape(X.shape[0], -1))
            gmm_models[speaker_id] = gmm
    
    return gmm_models

# 5. 识别说话人
def recognize_speaker(audio_file, gmm_models, feature_function):
    """
    识别给定音频的说话人
    
    Args:
        audio_file: 音频文件路径
        gmm_models: 包含所有说话人GMM模型的字典
        feature_function: 特征提取函数
        
    Returns:
        recognized_speaker: 识别出的说话人ID
        scores: 所有说话人的得分
    """
    # 提取特征
    features = feature_function(audio_file)
    
    # 计算每个说话人模型的得分
    scores = {}
    for speaker_id, gmm in gmm_models.items():
        scores[speaker_id] = gmm.score(features.reshape(1, -1))
    
    # 选择得分最高的说话人
    recognized_speaker = max(scores, key=scores.get)
    
    return recognized_speaker, scores

# 6. 评估模型
def evaluate_model(test_data, gmm_models, feature_function):
    """
    评估模型在测试集上的性能
    
    Args:
        test_data: 测试数据集
        gmm_models: 包含所有说话人GMM模型的字典
        feature_function: 特征提取函数
        
    Returns:
        accuracy: 准确率
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
    """
    true_labels = []
    pred_labels = []
    
    for speaker_id, audio_files in tqdm(test_data.items(), desc="评估模型"):
        for audio_file in audio_files:
            # 识别说话人
            recognized_speaker, _ = recognize_speaker(audio_file, gmm_models, feature_function)
            
            # 记录真实标签和预测标签
            true_labels.append(speaker_id)
            pred_labels.append(recognized_speaker)
    
    # 计算准确率
    accuracy = accuracy_score(true_labels, pred_labels)
    
    return accuracy, true_labels, pred_labels

# 主程序
if __name__ == "__main__":
    print("加载训练数据...")
    train_data, train_speakers = load_speaker_data(TrainDir)
    
    print("加载测试数据...")
    test_data, test_speakers = load_speaker_data(TestDir)
    
    # 特征提取参数 - 存储参数而不是函数
    mfcc_params = {"n_mfcc": 13, "n_fft": 2048, "hop_length": 512}
    
    # 特征提取函数 - 使用较少的MFCC系数
    feature_function = lambda x: extract_mfcc_features(x, **mfcc_params)
    
    print("提取训练数据特征...")
    train_features = extract_features_for_all_speakers(train_data, feature_function)
    
    print("训练GMM模型...")
    gmm_models = train_gmm_models(train_features, n_components=4, reg_covar=1e-1)  # 降低组件数量，增加正则化
    
    print("评估模型...")
    accuracy, true_labels, pred_labels = evaluate_model(test_data, gmm_models, feature_function)
    
    print(f"\n测试集准确率: {accuracy*100:.2f}%")
    
    # 绘制混淆矩阵 - 因为说话人太多，只绘制部分说话人
    speaker_subset = train_speakers[:30]  # 只取前30个说话人进行可视化
    # 筛选标签只包含这些说话人的样本
    subset_indices = [i for i, label in enumerate(true_labels) if label in speaker_subset]
    subset_true_labels = [true_labels[i] for i in subset_indices]
    subset_pred_labels = [pred_labels[i] for i in subset_indices]
    
    cm = confusion_matrix(subset_true_labels, subset_pred_labels, labels=speaker_subset)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=speaker_subset, yticklabels=speaker_subset)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('说话人识别混淆矩阵 (部分说话人)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    
    #TODO 保存模型
    
    
    # 对不同组件数量的GMM模型进行实验
    # components = [1, 2, 4, 8]  # 从更小的组件数量开始
    # accuracies = []
    
    # for n_components in components:
    #     print(f"\n训练 {n_components} 个组件的GMM模型...")
    #     gmm_models = train_gmm_models(train_features, n_components=n_components, reg_covar=1e-1)
        
    #     print("评估模型...")
    #     accuracy, _, _ = evaluate_model(test_data, gmm_models, feature_function)
    #     accuracies.append(accuracy)
        
    #     print(f"组件数量: {n_components}, 准确率: {accuracy*100:.2f}%")
    
    # 绘制不同组件数量的准确率比较图
    # plt.figure(figsize=(10, 6))
    # plt.plot(components, accuracies, 'bo-', linewidth=2)
    # plt.title('不同GMM组件数量的准确率比较')
    # plt.xlabel('GMM组件数量')
    # plt.ylabel('准确率')
    # plt.grid(True)
    # plt.savefig('gmm_components_comparison.png')
    
    # 实验不同特征提取方法
    print("\n比较不同特征提取方法...")
    
    # 定义不同的特征提取函数
    def extract_mfcc_delta_features(audio_file):
        """提取MFCC及其一阶和二阶差分特征"""
        audio, sr = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([np.mean(mfcc, axis=1), np.mean(delta, axis=1), np.mean(delta2, axis=1)])
        return combined.flatten()
    
    def extract_spectral_features(audio_file):
        """提取频谱特征"""
        audio, sr = librosa.load(audio_file, sr=None)
        # 提取短时傅里叶变换
        stft = np.abs(librosa.stft(audio))
        
        # 计算频谱质心
        cent = librosa.feature.spectral_centroid(S=stft, sr=sr)
        # 计算频谱平展度
        flat = librosa.feature.spectral_flatness(S=stft)
        # 计算频谱对比度
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        
        # 计算均值 - 修复不同维度的问题
        centroid_mean = np.mean(cent, axis=1)           # 形状为 (1,)
        flatness_mean = np.mean(flat, axis=1)           # 形状为 (1,)
        contrast_mean = np.mean(contrast, axis=1)       # 形状为 (7,) 对于部分音频
        
        # 将所有特征展平并连接
        features = np.concatenate([centroid_mean, flatness_mean, contrast_mean])
        
        return features
    
    # 定义要比较的特征提取方法
    feature_methods = {
        "MFCC": lambda x: extract_mfcc_features(x, **mfcc_params),
        "MFCC+Delta": extract_mfcc_delta_features,
        "Spectral": extract_spectral_features
    }
    
    # 比较不同特征提取方法
    feature_accuracies = {}
    
    for method_name, method_func in feature_methods.items():
        print(f"\n使用 {method_name} 特征...")
        
        print("提取训练数据特征...")
        train_features = extract_features_for_all_speakers(train_data, method_func)
        
        print("训练GMM模型...")
        gmm_models = train_gmm_models(train_features, n_components=4, reg_covar=1e-1)
        
        print("评估模型...")
        accuracy, _, _ = evaluate_model(test_data, gmm_models, method_func)
        feature_accuracies[method_name] = accuracy
        
        print(f"特征方法: {method_name}, 准确率: {accuracy*100:.2f}%")
    
    # 绘制不同特征方法的准确率比较图
    plt.figure(figsize=(10, 6))
    methods = list(feature_accuracies.keys())
    accs = [feature_accuracies[m] for m in methods]
    
    plt.bar(methods, accs)
    plt.title('不同特征提取方法的准确率比较')
    plt.xlabel('特征提取方法')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    
    # 在柱状图上添加准确率标签
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v*100:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('feature_methods_comparison.png')
    
    print("\n实验完成！")

