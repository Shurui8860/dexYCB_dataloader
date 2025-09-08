# 假设你已经设置了环境变量 'DEX_YCB_DIR' 来指向数据集的根目录

from dexYCB_toolkit.sequence_loader import DexYCBDataset

# 创建 DexYCBDataset 实例，选择适当的 setup 和 split
dataset = DexYCBDataset(setup='s0', split='train')  # 你可以选择其他的 setup 和 split，如 'val' 或 'test'

# 获取第一个样本的数据
sample = dataset[0]  # 样本 0

# 打印出数据样本中包含的内容
print("Sample keys:", sample.keys())
