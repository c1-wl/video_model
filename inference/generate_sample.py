import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from models import SimpleSingleStreamTransformer, VideoEncoder, AudioEncoder
import config

def main():
    d_model = config.D_MODEL
    T_video = config.T_VIDEO
    seq_len = T_video + 1
    device = config.DEVICE

    # 加载模型
    model = SimpleSingleStreamTransformer(
        d_model=d_model, nhead=config.NHEAD, num_layers=config.NUM_LAYERS
    ).to(device)
    video_enc = VideoEncoder(output_dim=d_model).to(device)
    audio_enc = AudioEncoder(input_length=config.AUDIO_LEN, output_dim=d_model).to(device)

    ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "checkpoint_last.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    video_enc.load_state_dict(ckpt['video_enc_state_dict'])
    audio_enc.load_state_dict(ckpt['audio_enc_state_dict'])

    model.eval()
    video_enc.eval()
    audio_enc.eval()

    sigma = 0.1
    z = torch.randn(1, seq_len, d_model, device=device)

    with torch.no_grad():
        pred_video, pred_audio = model(z)

    x_denoised = z - sigma * pred_video
    video_features = x_denoised[0, :T_video, :].cpu().numpy()
    audio_feature = x_denoised[0, T_video, :].cpu().numpy()

    # 可视化第一帧特征热图
    frame_feat = video_features[0, :64]
    img = frame_feat.reshape(8, 8)
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap='viridis', interpolation='nearest')
    plt.title("Model's imagined video frame (feature heatmap)")
    plt.colorbar()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "generated_frame_heatmap.png"))
    plt.close()
    print("已保存热图")

    # 生成测试音频（正弦波映射）
    sr = config.AUDIO_SAMPLE_RATE
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 200 + (audio_feature.mean() * 1800)
    freq = np.clip(freq, 200, 2000)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    sf.write(os.path.join(config.OUTPUT_DIR, "generated_audio.wav"), wave, sr)
    print("已保存音频")

if __name__ == "__main__":
    main()