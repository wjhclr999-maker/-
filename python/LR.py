# -*- coding: utf-8 -*-
"""
注意：本代码直接是.py 文件的内容, 你可以分段复制到jupyter(.ipynb)中运行，你也可以直接执行本.py 文件
本代码示例具有超级详细的注释, 如果依然有疑问, 优先问AI, 要养成习惯，这是新时代人类的习惯

实现一个简单的线性回归模型，使用 PyTorch 框架
目的是根据学生的信息预测其被大学研究生录取的概率
数据来源: Admission_Predict.csv (kaggle 数据集)
源作者: 依力 EL@zju.edu.cn

需要第三方库：
- pandas
一个用于数据处理和分析的库

- torch
PyTorch 框架，用于构建和训练深度学习模型

- matplotlib
用于绘图和数据可视化

- seaborn
用于数据可视化

- scikit-learn
一个用于专注于数据挖掘和数据分析的库，提供了许多机器学习算法和工具
"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# 英伟达 GPU 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Apple M系列 GPU 配置
# if torch.backends.mps.is_available():
#     device = torch.device("mps")  # 使用 Apple GPU
# else:
#     print("❌ MPS 不可用，回退到 CPU")
# device = torch.device("mps"  else "cpu")


# 设置支持中文字体（macOS是这样设置，Windows问一下AI，我没测试过）
# matplotlib.rcParams['font.sans-serif'] = ['PingFang HK', 'Heiti TC', 'Arial Unicode MS']  # 优先使用可用字体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# Windows系统适用的中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI Emoji','Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']  # Windows可用字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# 加载数据
file_path = "Admission_Predict.csv"  
df = pd.read_csv(file_path)

# 清洗列名
df.columns = [col.strip() for col in df.columns]

# 删除无用列
df.drop(columns=['Serial No.'], inplace=True)

for col in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 特征和目标变量
X = df.drop(columns=['Chance of Admit'])
y = df['Chance of Admit']

# 数据标准化
# 为什么要标准化？不同特征的量纲和取值范围可能不同，比如一个特征是 [0, 100]，另一个是 [0, 1]，这会导致模型训练不稳定或者收敛变慢
# 标准化让所有特征在相同尺度下，有助于模型：更快地收敛、更公平地使用每个特征、在梯度下降中保持合理步长
# 创建一个 StandardScaler 实例，StandardScaler 是 sklearn.preprocessing 模块中的一个工具，用于将特征缩放为均值为 0，标准差为 1 的分布，也称为 Z-score 标准化
# 它本身并不对数据做任何处理，仅创建一个准备进行缩放操作的“模型对象”
scaler = StandardScaler()
# 使用 fit_transform 方法对特征数据进行标准化，fit_transform 方法会先计算每个特征的均值和标准差，然后使用这些统计量对数据进行缩放
# StandardScaler().fit_transform(X) 的作用是将 X 中每一列（每个特征）转换为均值为0、方差为1的标准正态分布形式，为后续模型训练打好基础
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集，规则为 80% 训练集，20% 测试集
# train_test_split 是 sklearn.model_selection 模块中的一个函数，用于将数据集随机划分为训练集和测试集
# test_size=0.2 表示将 20% 的数据作为测试集，
# random_state=42 设置随机种子，确保每次运行时划分结果相同，这样可以保证实验的可复现性
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为张量
# PyTorch 中的张量是多维数组，类似于 NumPy 数组，但可以在 GPU 上进行高效计算
# torch.tensor() 将 NumPy 数组或其他数据结构转换为 PyTorch 张量
# dtype=torch.float32 指定数据类型为 32 位浮点数，适合深度学习模型的输入
# view(-1, 1) 将一维张量转换为二维张量，-1 表示自动计算行数，1 表示列数
# 这样做是为了确保标签 y_train 和 y_test 的形状符合模型的要求，通常模型期望输入为二维张量（样本数, 特征数），
# 但标签通常是一维的（样本数,），所以需要将其转换
# 这里的转换是为了确保模型在训练时能够正确处理标签数据
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 构建数据加载器
# batch_size每次迭代用多少个样本来计算梯度，小（如 1 或 16）学习更细致、泛化好，但训练慢、不稳定、噪声大；大（如 128 或 512）稳定、速度快但泛化能力差、内存占用大；32（经验值）稳定性 + 泛化折中好，是深度学习中常用默认值
# shuffle=True每个 epoch 随机打乱数据顺序，避免模型学到“样本顺序的偏见”，尤其对于梯度下降，如果每次 batch 的顺序一样，很容易陷入局部最优或过拟合，shuffle=True 每一轮都会重新洗牌，使得模型泛化能力更强
# 假设你是个老师，要教 60 个学生考试技巧：
# 如果你每次上课都让前 10 个是学霸，最后 10 个是学渣，学生就可能学会了“顺序”而不是“技巧”
# 你把学生洗牌，随机 10 人一组，每节课的学生都不一样，教学效果就更平均，模型泛化能力也更好
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_loader 测试集通常不需要打乱顺序，因为我们只关心最终的准确率，保持评估一致性，不需要“泛化训练”
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input_dim（x的元数）-> 1个y输出
        self.linear = nn.Linear(input_dim, 1)  # 线性层：y = Wx + b

    def forward(self, x):
        return self.linear(x)

# 创建一个线性回归模型实例，并告诉它输入的特征维度是多少
# X.shape[0] 是样本的数量，比如 400 行样本；X.shape[1] 是特征的数量，比如 7 个特征（GRE 分数、TOEFL 分数、大学排名、SOP 等）；
# 所以 X.shape[1] 其实就是模型输入的 维度（即特征个数）
model = LinearRegressionModel(input_dim=X.shape[1])


# 定义损失函数和优化器
# 定义损失函数（Loss Function）为均方误差损失（MSE, Mean Squared Error）
criterion = nn.MSELoss()


# 定义优化器（Optimizer）,随机梯度下降,学习率为 0.1，表示每次更新参数时走的“步长”，实际上用的是 小批量随机梯度下降，因为你传入的是小批量数据
# 0.1 是一个经验值，其实并没有一个放之四海而皆准的值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 想用Adam？可以试试
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 用于可视化的记录变量
train_losses = []

# 训练模型
epochs = 500
# 遍历每一个 epoch（训练周期）
for epoch in range(epochs):
    # 用于记录当前 epoch 的累计损失
    epoch_loss = 0.0 
    # 遍历训练数据，每次获取一个 batch（小批量）
    for batch_X, batch_y in train_loader:
        # 第一步：清空上一轮的梯度信息（避免累加）
        optimizer.zero_grad()
        # 第二步：将输入数据喂入模型，得到预测输出
        y_pred = model(batch_X)
        # 第三步：根据预测结果和真实值计算当前 batch 的损失
        loss = criterion(y_pred, batch_y)
        # 第四步：反向传播，计算每个参数的梯度
        loss.backward()
        # 第五步：使用优化器，根据梯度更新模型参数
        optimizer.step()
        # 累加当前 batch 的损失值
        epoch_loss += loss.item()
    
    # 计算当前 epoch 的平均损失值（所有 batch 的平均）
    avg_loss = epoch_loss / len(train_loader)
    # 记录损失值用于后续绘图
    train_losses.append(avg_loss)
    # 每训练 20 个 epoch 打印一次损失值
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 可视化训练损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# 模型评估：在测试集上进行预测
# 关闭 Dropout（随机丢弃部分神经元，防止过拟合）、BatchNorm（使用当前 batch 的均值和方差进行归一化） 等训练时特有的行为，确保推理时模型表现一致、稳定，模型进入评估模式
model.eval()
# 在推理或测试时禁用梯度计算，节省内存和计算资源，并加快速度（因为不需要反向传播）
with torch.no_grad():
    # 将测试集的特征 X_test_tensor 输入到你定义好的线性回归模型 model 中，得到预测结果
    # model(X_test_tensor) 表示前向传播，输出是一个 torch.Tensor，即模型预测的“录取概率”
    # .numpy() 是将 PyTorch 的张量（Tensor）转换为 NumPy 数组，方便后续用 Matplotlib 或 sklearn 的工具进行可视化、误差分析等处理
    predictions = model(X_test_tensor).numpy()
    # 将测试集中的真实标签 y_test_tensor（是 PyTorch 的张量）转换为 NumPy 数组，便于对比预测值和真实值
    # 同样也是为了配合后续评估（比如绘图、计算 MSE、R² 等）
    y_true = y_test_tensor.numpy()

# 绘制预测值 vs 实际值
plt.figure(figsize=(8, 6))
plt.scatter(y_true, predictions, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', label='Ideal Prediction')
plt.xlabel("True Chance of Admit")
plt.ylabel("Predicted Chance of Admit")
plt.title("Prediction vs. True Values")
plt.legend()
plt.grid(True)
plt.show()

# 残差分析
residuals = y_true - predictions
plt.figure(figsize=(8, 5))
sns.histplot(residuals.flatten(), bins=30, kde=True)
plt.title("Distribution of Residuals (True - Predicted)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 模型评价指标
mse = mean_squared_error(y_true, predictions)
r2 = r2_score(y_true, predictions)
print(f"Test MSE: {mse:.4f}")
print(f"Test R^2 Score: {r2:.4f}")

# 展示模型权重（可解释性）
weights = model.linear.weight.detach().numpy().flatten()
feature_importance = pd.Series(weights, index=X.columns)
feature_importance.plot(kind='barh', figsize=(10, 6), title="Feature Weights in Linear Model")
plt.grid(True)
plt.show()

# 保存权重文件
torch.save(model.state_dict(), "linear_regression_model.pt")
