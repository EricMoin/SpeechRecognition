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


def extract_mfcc_features(file_path, n_mfcc=20, n_fft=1024, hop_length=256):
    """从音频文件中提取MFCC特征及其delta和delta-delta"""
    try:
        waveform, sample_rate = librosa.load(file_path, sr=16000)

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

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # 返回一个空数组，后续会在数据集中处理这个错误
        return np.array([])

# 创建数据集类


class SpeakerDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()

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
        features = extract_mfcc_features(file_path)

        # 处理提取失败的情况
        if features.size == 0:
            # 如果提取失败，返回一个全零特征
            # 60 = 3 * 20 (MFCC + delta + delta2)
            features = np.zeros((60, 300))

        # 确保所有特征大小一致，通过填充或截断
        target_length = 300  # 设置一个固定长度
        if features.shape[1] < target_length:
            pad_width = target_length - features.shape[1]
            features = np.pad(
                features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :target_length]

        # 标准化
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        # 增加通道维度，并转换为PyTorch张量
        features = torch.FloatTensor(
            features).unsqueeze(0)  # [1, 3*n_mfcc, time]
        label = torch.tensor(label, dtype=torch.long)

        return features, label

# 简化的ResNet模型


class SimplerResNetSpeaker(nn.Module):
    def __init__(self, num_classes):
        super(SimplerResNetSpeaker, self).__init__()

        # 基本的卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ResNet风格的块
        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)

        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        # 第一个块可能改变通道数
        layers.append(ResBlock(in_channels, out_channels, stride=1))
        # 其余块保持通道数不变
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ResNet基本块


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道数不同，添加一个投影捷径
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

# 训练模型


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=7):
    writer = SummaryWriter('runs/speaker_recognition_resnet')

    since = time.time()
    best_acc = 0.0
    best_model_weights = None

    # 早停机制
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Training")

        for inputs, labels in progress_bar:
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
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            progress_bar.set_postfix({"loss": loss.item(), "acc": torch.sum(
                preds == labels.data).item() / inputs.size(0)})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        writer.add_scalar('Loss/validation', epoch_loss, epoch)
        writer.add_scalar('Accuracy/validation', epoch_acc, epoch)

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, 'best_speaker_model.pth')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 早停
        if no_improve_epochs >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    return model

# 评估模型函数


def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    return test_acc, all_preds, all_labels

# 绘制混淆矩阵


def plot_confusion_matrix(y_true, y_pred, classes, top_n=30):
    """绘制前top_n个最常见说话人的混淆矩阵"""
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

    # 计算每个说话人的准确率
    per_class_accuracy = {}
    for i, class_idx in enumerate(top_classes_idx):
        class_true = np.where(filtered_y_true == class_idx)[0]
        if len(class_true) > 0:
            correct = np.sum(filtered_y_pred[class_true] == class_idx)
            per_class_accuracy[classes[class_idx]] = correct / len(class_true)

    # 绘制每个说话人的准确率
    plt.figure(figsize=(15, 8))
    speakers = list(per_class_accuracy.keys())
    accuracies = list(per_class_accuracy.values())

    # 按准确率排序
    sorted_indices = np.argsort(accuracies)[::-1]
    speakers = [speakers[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    plt.bar(speakers, accuracies)
    plt.xlabel('说话人')
    plt.ylabel('准确率')
    plt.title('每个说话人的识别准确率')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('per_speaker_accuracy.png', dpi=300)
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
        train_subset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=64,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, num_workers=2)

    # 创建简化的ResNet模型实例
    model = SimplerResNetSpeaker(num_classes=num_classes)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 对于平衡数据集使用标准交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # 训练模型
    print("开始训练模型...")
    model = train_model(model, train_loader, val_loader, criterion,
                        optimizer, scheduler, num_epochs=50, patience=7)

    # 评估模型
    print("在测试集上评估模型...")
    test_acc, all_preds, all_labels = evaluate_model(
        model, test_loader, criterion)

    # 绘制混淆矩阵和每个说话人的准确率
    plot_confusion_matrix(all_labels, all_preds, speaker_classes, top_n=20)

    # 保存结果
    results = {
        'accuracy': test_acc.item(),
        'predictions': all_preds,
        'true_labels': all_labels,
        'speaker_classes': speaker_classes
    }
    with open('resnet_speaker_recognition_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"最终测试准确率: {test_acc:.4f}")
    print("模型训练和评估完成，结果已保存。")


if __name__ == "__main__":
    main()
