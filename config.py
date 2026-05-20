# config.py
import torch

# 路径
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

# 模型参数
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
MAX_SEQ_LEN = 256

# 数据参数
VIDEO_SIZE = (64, 64)
FPS = 8
AUDIO_SAMPLE_RATE = 16000
DURATION = 1.0
T_VIDEO = int(DURATION * FPS)          # 8
AUDIO_LEN = int(DURATION * AUDIO_SAMPLE_RATE)  # 16000

# 训练参数
BATCH_SIZE = 1
LR = 1e-3
NUM_EPOCHS_DUMMY = 100
NUM_EPOCHS_DIFFUSION = 200
NUM_EPOCHS_DECODER = 100

# 扩散调度器参数
NUM_TIMESTEPS = 100
BETA_START = 1e-4
BETA_END = 0.02

# 设备
DEVICE = torch.device("cpu")   # 可按需改为 cuda


# ========== 条件生成相关配置 ==========
TEXT_EMBED_DIM = 128          # 文本嵌入维度
IMAGE_EMBED_DIM = 128         # 图像嵌入维度
USE_CLIP = False               # 是否使用CLIP（需要安装transformers）
MAX_TEXT_LEN = 32              # 文本最大token数
VOCAB_SIZE = 1000              # 简单词表大小（仅当USE_CLIP=False时使用）

# 条件训练
NUM_EPOCHS_COND = 200
COND_LR = 1e-3