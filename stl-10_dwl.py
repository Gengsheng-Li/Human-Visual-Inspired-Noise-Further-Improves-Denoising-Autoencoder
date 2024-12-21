import torchvision
import torchvision.transforms as transforms

# 设置数据集保存路径
root = 'E:\Dataset\STL10'

# 下载训练集
trainset = torchvision.datasets.STL10(root=root, split='train', download=True)

# 下载测试集
testset = torchvision.datasets.STL10(root=root, split='test', download=True)

print("STL-10 dataset downloaded successfully!")