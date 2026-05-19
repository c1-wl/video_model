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