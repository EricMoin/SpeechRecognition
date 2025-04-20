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
import tqdm
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
import hashlib

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

# 定义提取MFCC特征的函数


def extract_mfcc_features(file_path, n_mfcc=40, n_fft=512, hop_length=160, use_cache=True):
    """从音频文件中提取MFCC特征"""
    # 为文件路径和参数创建唯一的缓存文件名
    if use_cache:
        params_str = f"mfcc{n_mfcc}_fft{n_fft}_hop{hop_length}"
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        cache_filename = os.path.join(
            FEATURE_CACHE_DIR, f"{file_hash}_{params_str}.npz")

        # 检查缓存是否存在
        if os.path.exists(cache_filename):
            try:
                cached_data = np.load(cache_filename)
                features = cached_data['features']
                return features
            except Exception as e:
                print(f"Error loading cached features for {file_path}: {e}")

    # 如果没有缓存或加载缓存失败，提取特征
    try:
        waveform, sample_rate = librosa.load(file_path, sr=16000)

        # 预加重以增强高频部分
        waveform = librosa.effects.preemphasis(waveform, coef=0.97)

        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # 添加delta和delta-delta特征
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # 将三者叠加在一起
        features = np.concatenate(
            [mfcc, delta_mfcc, delta2_mfcc], axis=0)  # [3*n_mfcc, time]

        # 确保有足够的帧数
        target_length = 300
        if features.shape[1] < target_length:
            pad_width = target_length - features.shape[1]
            features = np.pad(
                features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :target_length]

        # 保存到缓存
        if use_cache:
            try:
                np.savez_compressed(cache_filename, features=features)
            except Exception as e:
                print(f"Error saving features to cache for {file_path}: {e}")

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.array([])

# 创建数据集类


class SpeakerDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        # 特征提取参数
        self.n_mfcc = 40
        self.n_fft = 2048
        self.hop_length = 512
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

        # 提取MFCC特征
        features = extract_mfcc_features(
            file_path,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 处理提取失败的情况
        if features.size == 0:
            features = np.zeros((3 * self.n_mfcc, 300))

        # 应用归一化
        features_mean = np.mean(features, axis=1, keepdims=True)
        features_std = np.std(features, axis=1, keepdims=True) + 1e-8
        features = (features - features_mean) / features_std

        # 数据增强（仅在训练集上）
        if self.is_train and random.random() < 0.3:
            # 添加随机噪声
            noise_level = random.uniform(0.0, 0.1)
            noise = np.random.randn(*features.shape) * noise_level
            features = features + noise

            # 随机屏蔽一些时间帧
            if random.random() < 0.5:
                time_mask_size = random.randint(5, 50)
                start_frame = random.randint(
                    0, features.shape[1] - time_mask_size)
                features[:, start_frame:start_frame + time_mask_size] = 0

        # 调整维度以适应ResNet输入 - 转换为3通道图像格式
        # 将特征图重复3次以匹配ResNet的3通道输入
        features = np.repeat(features[np.newaxis, :, :], 3, axis=0)
        features = torch.FloatTensor(features)  # [3, 3*n_mfcc, time]
        label = torch.tensor(label, dtype=torch.long)

        return features, label

# 定义ResNet模型


class SpeakerResNet(nn.Module):
    def __init__(self, num_classes):
        super(SpeakerResNet, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 修改第一层卷积层以接受我们的输入
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改最后的全连接层以适应我们的分类任务
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 评估模型函数


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="Testing"):
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
        train_subset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=32,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=2)

    # 获取输入特征维度
    sample_data, _ = train_dataset[0]
    print(f"样本数据形状: {sample_data.shape}")

    # 创建ResNet模型
    model = SpeakerResNet(num_classes=num_classes)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6)

    # 训练模型
    print("开始训练模型...")
    num_epochs = 30
    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_total = 0

        progress_bar = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
            progress_bar.set_postfix({"loss": loss.item(), "acc": (
                torch.sum(preds == labels.data).item() / inputs.size(0))})

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

    # 保存结果
    results = {
        'accuracy': test_acc,
        'predictions': all_preds,
        'true_labels': all_labels,
        'speaker_classes': speaker_classes,
    }

    with open('speaker_recognition_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"最终测试准确率: {test_acc:.4f}")
    print("模型训练和评估完成，结果已保存。")


if __name__ == "__main__":
    main()
