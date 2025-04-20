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

# 导入必要的库
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.preprocessing import LabelEncoder
import random
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import hashlib
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# 可以根据需要导入其他库，比如librosa用于音频处理
# 数据集基本信息如下
# 方言地区：DR1～DR8
# 性别：F/M
# 说话者ID：3个大写字母+1个阿拉伯数字
# 句子ID：句子类型（SA/SI/SX）+编号

TrainDir = "Dataset/TRAIN"
TestDir = "Dataset/TEST"

# 创建特征缓存目录
FEATURE_CACHE_DIR = "feature_cache"
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# 设置随机种子，确保结果可复现


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义音频增强函数


def augment_audio(waveform, sample_rate=16000):
    """应用音频增强技术"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    augment_type = random.randint(0, 4)

    if augment_type == 0:
        # 添加高斯噪声
        noise_level = 0.005
        noise = np.random.randn(*waveform.shape) * noise_level
        waveform = waveform + noise
    elif augment_type == 1:
        # 时间拉伸 (0.9-1.1)
        stretch_factor = random.uniform(0.9, 1.1)
        waveform = librosa.effects.time_stretch(waveform, rate=stretch_factor)
        # 确保长度一致
        if len(waveform) > sample_rate * 3:
            waveform = waveform[:sample_rate * 3]
        else:
            waveform = np.pad(
                waveform, (0, max(0, sample_rate * 3 - len(waveform))))
    elif augment_type == 2:
        # 音高变化 (-2到2半音)
        pitch_shift = random.randint(-2, 2)
        waveform = librosa.effects.pitch_shift(
            waveform, sr=sample_rate, n_steps=pitch_shift)
    elif augment_type == 3:
        # 随机静音片段
        silence_length = int(random.uniform(0.05, 0.15) * waveform.shape[0])
        start_idx = random.randint(0, waveform.shape[0] - silence_length)
        waveform[start_idx:start_idx + silence_length] = 0

    return waveform

# 定义提取MFCC特征的函数

# mfcc_params = {"n_mfcc": 40, "n_fft": 2048, "hop_length": 512}
# 这是最佳的参数设置


def extract_mfcc_features(file_path, n_mfcc=40, n_fft=512, hop_length=160, use_cache=True):
    """从音频文件中提取MFCC特征及其delta和delta-delta，支持缓存"""
    # 为文件路径和参数创建唯一的缓存文件名
    if use_cache:
        # 创建参数字符串用于缓存文件名
        params_str = f"mfcc{n_mfcc}_fft{n_fft}_hop{hop_length}"
        # 使用哈希来缩短文件名长度，避免路径过长问题
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        cache_filename = os.path.join(
            FEATURE_CACHE_DIR, f"{file_hash}_{params_str}.npz")

        # 检查缓存是否存在
        if os.path.exists(cache_filename):
            try:
                # 从缓存加载特征
                cached_data = np.load(cache_filename)
                features = cached_data['features']
                return features
            except Exception as e:
                print(f"Error loading cached features for {file_path}: {e}")
                # 如果加载缓存失败，继续提取特征

    # 如果没有缓存或加载缓存失败，提取特征
    try:
        waveform, sample_rate = librosa.load(file_path, sr=16000)

        # 预加重以增强高频部分 (用于x-vector系统)
        waveform = librosa.effects.preemphasis(waveform, coef=0.97)

        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            # 对于x-vector通常使用23个梅尔滤波器组，覆盖20-7600Hz频率范围
            htk=True,  # 使用HTK风格的梅尔频率
            n_mels=23
        )

        # 添加delta和delta-delta特征
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # 将三者叠加在一起
        features = np.concatenate(
            [mfcc, delta_mfcc, delta2_mfcc], axis=0)  # [3*n_mfcc, time]

        # 应用语音活动检测(VAD)，过滤静音帧
        # 简单的基于能量的VAD
        energy = np.sqrt(np.sum(mfcc**2, axis=0))
        threshold = np.mean(energy) * 0.5
        speech_frames = energy > threshold

        # 如果至少有30%的帧被判断为语音，则使用VAD过滤
        if np.mean(speech_frames) > 0.3:
            features = features[:, speech_frames]

        # 确保有足够的帧数
        if features.shape[1] < 50:
            # 如果帧数不足，复制现有帧直到达到所需数量
            repeat_times = int(np.ceil(50 / features.shape[1]))
            features = np.tile(features, (1, repeat_times))
            features = features[:, :50]  # 确保不超过50帧

        # 保存到缓存
        if use_cache:
            try:
                np.savez_compressed(cache_filename, features=features)
            except Exception as e:
                print(f"Error saving features to cache for {file_path}: {e}")

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # 返回一个空数组，后续会在数据集中处理这个错误
        return np.array([])

# 创建数据集类


class SpeakerDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        # 特征提取参数 - 优化用于x-vector
        self.n_mfcc = 23  # 使用与n_mels相同的值
        self.n_fft = 512  # 更小的FFT大小，适合TDNN
        self.hop_length = 160  # 10ms的帧移，符合x-vector的推荐设置

        # 收集所有话者和其对应的音频文件
        self._collect_data()

    def _collect_data(self):
        """收集所有音频文件和对应的说话人标签"""
        speaker_dirs = []
        for dialect in os.listdir(self.root_dir):
            dialect_path = os.path.join(self.root_dir, dialect)
            if os.path.isdir(dialect_path):
                for speaker in os.listdir(dialect_path):
                    speaker_path = os.path.join(dialect_path, speaker)
                    if os.path.isdir(speaker_path):
                        speaker_dirs.append((speaker, speaker_path))

        # 为每个说话人收集音频文件
        for speaker, speaker_path in speaker_dirs:
            audio_files = [f for f in os.listdir(speaker_path) if f.endswith(
                '.wav') and f != 'merge_result.wav']

            # 添加所有音频文件和对应标签
            for audio_file in audio_files:
                file_path = os.path.join(speaker_path, audio_file)
                self.file_paths.append(file_path)
                self.labels.append(speaker)

        # 编码标签
        self.label_encoder.fit(self.labels)
        self.labels = self.label_encoder.transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"找到{self.num_classes}个说话人, {len(self.file_paths)}个音频文件")

        # 保存标签编码器
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 提取MFCC特征及其delta和delta-delta
        features = extract_mfcc_features(
            file_path,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 处理提取失败的情况
        if features.size == 0:
            # 如果提取失败，返回一个全零特征
            features = np.zeros((3 * self.n_mfcc, 300))

        # 确保所有特征大小一致，通过填充或截断
        target_length = 300  # 设置一个固定长度
        if features.shape[1] < target_length:
            pad_width = target_length - features.shape[1]
            features = np.pad(
                features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :target_length]

        # 应用Cepstral Mean Variance Normalization (CMVN) - 常用于x-vector系统
        features_mean = np.mean(features, axis=1, keepdims=True)
        features_std = np.std(features, axis=1, keepdims=True) + 1e-8
        features = (features - features_mean) / features_std

        # 数据增强（仅在训练集上）
        if self.is_train and random.random() < 0.3:
            # 添加随机噪声
            noise_level = random.uniform(0.0, 0.1)
            noise = np.random.randn(*features.shape) * noise_level
            features = features + noise

            # 随机屏蔽一些时间帧(类似SpecAugment)
            if random.random() < 0.5:
                time_mask_size = random.randint(5, 50)
                start_frame = random.randint(
                    0, features.shape[1] - time_mask_size)
                features[:, start_frame:start_frame + time_mask_size] = 0

            # 随机屏蔽一些特征维度
            if random.random() < 0.5:
                freq_mask_size = random.randint(5, 15)
                start_dim = random.randint(
                    0, features.shape[0] - freq_mask_size)
                features[start_dim:start_dim + freq_mask_size, :] = 0

        # 增加通道维度，并转换为PyTorch张量
        features = torch.FloatTensor(
            features).unsqueeze(0)  # [1, 3*n_mfcc, time]
        label = torch.tensor(label, dtype=torch.long)

        return features, label


class TDNN(nn.Module):
    """
    TDNN层，用于x-vector模型
    context: 上下文窗口大小[左边界, 右边界]
    """

    def __init__(self, input_dim, output_dim, context, dilation=1):
        super(TDNN, self).__init__()
        self.context = context
        self.kernel_size = context[1] - context[0] + 1
        self.dilation = dilation

        # 计算需要的填充量，使得输出与输入时间维度相同
        padding = 0
        if context[0] < 0:
            padding += abs(context[0]) * dilation
        if context[1] > 0:
            padding += context[1] * dilation

        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,  # 使用计算的填充量
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x的输入形状为[batch, feature_dim, time]
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class XVectorModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(XVectorModel, self).__init__()

        # 帧级处理 - TDNN层
        self.frame1 = TDNN(input_dim, 512, context=[-2, 2])
        self.frame2 = TDNN(512, 512, context=[-2, 1])
        self.frame3 = TDNN(512, 512, context=[-3, 1])
        self.frame4 = TDNN(512, 512, context=[-1, 1])
        self.frame5 = TDNN(512, 1500, context=[0, 0])

        # 统计池化层
        self.stats_pooling = StatsPooling()

        # 段级处理 - 全连接层
        self.segment1 = nn.Linear(3000, 512)
        self.segment_bn1 = nn.BatchNorm1d(512)
        self.segment_relu1 = nn.ReLU()

        self.segment2 = nn.Linear(512, 512)
        self.segment_bn2 = nn.BatchNorm1d(512)
        self.segment_relu2 = nn.ReLU()

        # 输出层
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        # 输入x形状：[batch, 1, feature_dim, time]
        # 调整形状以适应TDNN
        x = x.squeeze(1)  # [batch, feature_dim, time]

        # 帧级处理
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        # 统计池化
        x = self.stats_pooling(x)

        # 段级处理
        x = self.segment1(x)
        x = self.segment_bn1(x)
        x = self.segment_relu1(x)

        x = self.segment2(x)
        x = self.segment_bn2(x)
        x = self.segment_relu2(x)

        # 输出层
        x = self.output(x)

        return x

    def extract_xvector(self, x):
        """提取x-vector嵌入表示"""
        # 输入x形状：[batch, 1, feature_dim, time]
        # 调整形状以适应TDNN
        x = x.squeeze(1)  # [batch, feature_dim, time]

        # 帧级处理
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        # 统计池化
        x = self.stats_pooling(x)

        # 段级处理 - 提取embedding
        x = self.segment1(x)
        x = self.segment_bn1(x)
        x = self.segment_relu1(x)

        return x  # 返回512维的x-vector


class StatsPooling(nn.Module):
    """统计池化层，计算所有帧的均值和标准差"""

    def __init__(self):
        super(StatsPooling, self).__init__()

    def forward(self, x):
        # x的形状: [batch, feature_dim, time]
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)

        # 连接均值和标准差
        pooled = torch.cat((mean, std), dim=1)  # [batch, 2*feature_dim]
        return pooled

# 评估模型函数


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {test_acc:.4f}')

    return test_acc, all_preds, all_labels

# 绘制混淆矩阵


def plot_confusion_matrix(y_true, y_pred, classes, top_n=20):
    # 计算每个说话人在测试集中的出现次数
    class_counts = {}
    for label in y_true:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # 获取出现次数最多的top_n个说话人
    top_classes_idx = sorted(
        class_counts.keys(), key=lambda x: class_counts[x], reverse=True)[:top_n]

    # 筛选这些说话人的预测结果
    mask = np.isin(y_true, top_classes_idx)
    filtered_y_true = np.array(y_true)[mask]
    filtered_y_pred = np.array(y_pred)[mask]

    # 获取这些说话人的类别名称
    top_classes = [classes[idx] for idx in top_classes_idx]

    # 计算混淆矩阵
    cm = confusion_matrix(filtered_y_true, filtered_y_pred,
                          labels=top_classes_idx)

    # 绘制混淆矩阵
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_classes, yticklabels=top_classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'说话人识别混淆矩阵 (Top {top_n})')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_xvectors(model, data_loader, speaker_classes, max_samples=1000):
    """
    提取x-vector嵌入并使用t-SNE进行可视化
    """
    model.eval()
    all_xvectors = []
    all_labels = []
    sample_count = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="提取x-vectors"):
            inputs = inputs.to(device)

            # 提取x-vector嵌入表示
            xvectors = model.extract_xvector(inputs)

            all_xvectors.append(xvectors.cpu().numpy())
            all_labels.extend(labels.numpy())

            sample_count += inputs.size(0)
            if sample_count >= max_samples:
                break

    # 合并所有x-vector
    all_xvectors = np.vstack(all_xvectors)
    all_labels = np.array(all_labels)

    # 使用t-SNE降维到2D进行可视化
    print("使用t-SNE进行降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    xvectors_2d = tsne.fit_transform(all_xvectors)

    # 随机选择最多20个说话人进行可视化
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 20:
        selected_labels = np.random.choice(unique_labels, 20, replace=False)
        mask = np.isin(all_labels, selected_labels)
        xvectors_2d = xvectors_2d[mask]
        all_labels = all_labels[mask]

    # 绘制t-SNE可视化
    plt.figure(figsize=(12, 10))

    for label in np.unique(all_labels):
        indices = np.where(all_labels == label)
        plt.scatter(
            xvectors_2d[indices, 0],
            xvectors_2d[indices, 1],
            label=speaker_classes[label],
            alpha=0.7
        )

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('说话人 x-vector 嵌入 (t-SNE)')
    plt.savefig('xvector_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("x-vector可视化已保存到 'xvector_tsne.png'")


def demonstrate_speaker_verification(model, test_loader, speaker_classes, num_pairs=100):
    """
    使用x-vector嵌入演示说话人验证功能
    """
    from sklearn.metrics import roc_curve, auc

    model.eval()

    # 按说话人收集样本
    speaker_samples = {}
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="收集验证样本"):
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()

            # 提取x-vector嵌入
            xvectors = model.extract_xvector(inputs).cpu().numpy()

            # 按说话人整理样本
            for i in range(len(labels)):
                speaker_id = labels[i]
                if speaker_id not in speaker_samples:
                    speaker_samples[speaker_id] = []
                speaker_samples[speaker_id].append(xvectors[i])

    # 仅保留有足够样本的说话人(至少2个样本)
    speaker_samples = {k: v for k, v in speaker_samples.items() if len(v) >= 2}

    if len(speaker_samples) < 2:
        print("没有足够的说话人样本进行验证，至少需要2个说话人，每个说话人至少2个样本")
        return None, None

    # 生成正样本对(相同说话人)和负样本对(不同说话人)
    same_speaker_pairs = []  # (嵌入1, 嵌入2, 标签1=1)
    different_speaker_pairs = []  # (嵌入1, 嵌入2, 标签0=0)

    # 生成正样本对
    for speaker_id, embeddings in speaker_samples.items():
        if len(embeddings) < 2:
            continue

        # 从这个说话人的样本中随机选择pair_count对
        pair_count = min(num_pairs // len(speaker_samples),
                         len(embeddings) * (len(embeddings) - 1) // 2)
        pair_count = max(1, pair_count)  # 确保至少有一对

        indices = list(range(len(embeddings)))

        for _ in range(pair_count):
            i, j = random.sample(indices, 2)
            same_speaker_pairs.append((embeddings[i], embeddings[j], 1))

    # 生成负样本对
    speaker_ids = list(speaker_samples.keys())
    if len(speaker_ids) >= 2:  # 确保有至少两个说话人
        for _ in range(min(num_pairs, len(same_speaker_pairs))):
            # 随机选择两个不同的说话人
            speaker1, speaker2 = random.sample(speaker_ids, 2)

            # 从各自的样本中随机选择一个嵌入向量
            embedding1 = random.choice(speaker_samples[speaker1])
            embedding2 = random.choice(speaker_samples[speaker2])

            different_speaker_pairs.append((embedding1, embedding2, 0))

    # 检查是否有足够的样本对
    if len(same_speaker_pairs) == 0 or len(different_speaker_pairs) == 0:
        print("没有足够的样本对进行验证")
        return None, None

    # 合并所有样本对
    all_pairs = same_speaker_pairs + different_speaker_pairs
    random.shuffle(all_pairs)

    # 计算余弦相似度和真实标签
    similarities = []
    true_labels = []

    for embedding1, embedding2, label in all_pairs:
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / \
            (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarities.append(similarity)
        true_labels.append(label)

    # 检查是否有足够的正负样本
    if 1 not in true_labels or 0 not in true_labels:
        print("错误：缺少正样本或负样本，无法计算ROC曲线")
        return None, None

    try:
        # 绘制相似度分布
        plt.figure(figsize=(10, 6))

        # 获取正样本和负样本的相似度
        positive_similarities = [similarities[i] for i in range(
            len(similarities)) if true_labels[i] == 1]
        negative_similarities = [similarities[i] for i in range(
            len(similarities)) if true_labels[i] == 0]

        plt.hist(positive_similarities, bins=30,
                 alpha=0.5, label='相同说话人', color='green')
        plt.hist(negative_similarities, bins=30,
                 alpha=0.5, label='不同说话人', color='red')

        plt.xlabel('余弦相似度')
        plt.ylabel('频率')
        plt.title('说话人验证相似度分布')
        plt.legend()
        plt.savefig('similarity_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(true_labels, similarities)
        roc_auc = auc(fpr, tpr)

        # 找出等错误率点(EER)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer_threshold = thresholds[eer_threshold_idx]
        eer = fpr[eer_threshold_idx]

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'说话人验证ROC曲线 (EER = {eer:.2f})')
        plt.legend(loc="lower right")
        plt.savefig('speaker_verification_roc.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制带阈值的相似度分布
        plt.figure(figsize=(10, 6))
        plt.hist(positive_similarities, bins=30,
                 alpha=0.5, label='相同说话人', color='green')
        plt.hist(negative_similarities, bins=30,
                 alpha=0.5, label='不同说话人', color='red')
        plt.axvline(x=eer_threshold, color='black', linestyle='--',
                    label=f'EER阈值 ({eer_threshold:.2f})')
        plt.xlabel('余弦相似度')
        plt.ylabel('频率')
        plt.title('说话人验证相似度分布')
        plt.legend()
        plt.savefig('similarity_distribution_with_threshold.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"说话人验证结果: AUC = {roc_auc:.4f}, EER = {eer:.4f}")
        print(f"相似度分布和ROC曲线已保存")

        return roc_auc, eer

    except Exception as e:
        print(f"绘制曲线时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 主函数


def main():

    # 创建数据集和数据加载器
    train_dataset = SpeakerDataset(TrainDir, is_train=True)
    test_dataset = SpeakerDataset(TestDir, is_train=False)

    # 获取说话人类别映射
    speaker_classes = train_dataset.label_encoder.classes_
    num_classes = len(speaker_classes)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_subset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=64,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, num_workers=2)

    # 获取输入特征维度
    sample_data, _ = train_dataset[0]
    print(f"样本数据形状: {sample_data.shape}")  # 应为 [1, input_dim, time]
    # [1, feature_dim, time] -> feature_dim
    input_dim = sample_data.squeeze(0).shape[0]
    print(f"输入特征维度: {input_dim}")

    # 创建TDNN + x-vector模型
    model = XVectorModel(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    # 检查模型第一层的期望输入维度
    conv_in_channels = model.frame1.conv.in_channels
    print(f"模型第一层卷积输入通道数: {conv_in_channels}")
    assert conv_in_channels == input_dim, f"输入维度不匹配: 模型期望 {conv_in_channels}, 实际输入 {input_dim}"

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6)

    # 训练模型
    print("开始训练模型...")
    num_epochs = 40
    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (inputs, labels) in enumerate(progress_bar):
            # 在第一个批次中打印输入形状，确保维度正确
            if epoch == 0 and i == 0:
                print(f"批次输入形状: {inputs.shape}")

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            train_total += inputs.size(0)

            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item(),
                                      "acc": (torch.sum(preds == labels.data).item() / inputs.size(0))})

        epoch_loss = running_loss / train_total
        epoch_acc = running_corrects.double() / train_total
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        epoch_acc = running_corrects.double() / val_total
        print(f'Val Acc: {epoch_acc:.4f}')

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_model_weights = model.state_dict().copy()

    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, 'best_speaker_model.pth')

    # 在测试集上评估模型
    print("在测试集上评估模型...")
    test_acc, all_preds, all_labels = evaluate_model(model, test_loader)

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, speaker_classes)

    # 可视化x-vector表示
    print("提取并可视化x-vector嵌入...")
    visualize_xvectors(model, test_loader, speaker_classes)

    # 演示说话人验证功能
    print("演示说话人验证功能...")
    verification_results = demonstrate_speaker_verification(
        model, test_loader, speaker_classes)

    # 保存结果
    results = {
        'accuracy': test_acc,
        'predictions': all_preds,
        'true_labels': all_labels,
        'speaker_classes': speaker_classes,
    }

    # 如果验证成功，添加验证结果
    if verification_results and verification_results[0] is not None:
        roc_auc, eer = verification_results
        results.update({
            'verification_auc': roc_auc,
            'verification_eer': eer
        })

    with open('speaker_recognition_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"最终测试准确率: {test_acc:.4f}")
    print("模型训练和评估完成，结果已保存。")


if __name__ == "__main__":
    main()
