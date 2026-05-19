import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import imageio
import soundfile as sf
import subprocess
from pathlib import Path
from models import SimpleSingleStreamTransformer, VideoEncoder, AudioEncoder, VideoDecoder, AudioDecoder, SimpleDDPMScheduler
import config

def main():
    device = config.DEVICE
    d_model = config.D_MODEL
    T_video = config.T_VIDEO
    seq_len = T_video + 1

    # 加载扩散模型和编码器
    diff_ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "diffusion_checkpoint.pth"), map_location=device)
    video_enc = VideoEncoder(output_dim=d_model).to(device)
    audio_enc = AudioEncoder(input_length=config.AUDIO_LEN, output_dim=d_model).to(device)
    model = SimpleSingleStreamTransformer(d_model=d_model, nhead=config.NHEAD, num_layers=config.NUM_LAYERS).to(device)
    video_enc.load_state_dict(diff_ckpt['video_enc_state'])
    audio_enc.load_state_dict(diff_ckpt['audio_enc_state'])
    model.load_state_dict(diff_ckpt['model_state'])
    video_enc.eval(); audio_enc.eval(); model.eval()

    # 加载解码器
    dec_ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "decoders.pth"), map_location=device)
    video_dec = VideoDecoder(input_dim=d_model).to(device)
    audio_dec = AudioDecoder(input_dim=d_model, output_length=config.AUDIO_LEN).to(device)
    video_dec.load_state_dict(dec_ckpt['video_dec_state'])
    audio_dec.load_state_dict(dec_ckpt['audio_dec_state'])
    video_dec.eval(); audio_dec.eval()

    # 调度器
    scheduler = SimpleDDPMScheduler(
        num_timesteps=config.NUM_TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    # 恢复 betas（保险起见从 checkpoint 中读取）
    betas = diff_ckpt['scheduler_config']['betas'].to(device)
    alphas = 1.0 - betas
    scheduler.betas = betas
    scheduler.alphas = alphas
    scheduler.alpha_bars = torch.cumprod(alphas, dim=0)

    # 从纯噪声开始采样
    x = torch.randn(1, seq_len, d_model, device=device)
    num_steps = config.NUM_TIMESTEPS
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_video, _ = model(x)   # 只用 video 头预测噪声
        alpha_bar = scheduler.alpha_bars[t]
        beta_t = scheduler.betas[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        # 去噪公式
        x = (1 / sqrt_alpha_bar) * (x - (beta_t / torch.sqrt(1 - alpha_bar)) * pred_video)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise

    # 分离特征
    video_features = x[0, :T_video, :].unsqueeze(0)   # (1, T, D)
    audio_feature = x[0, T_video, :].unsqueeze(0).unsqueeze(0)  # (1, 1, D)

    # 解码视频
    with torch.no_grad():
        frames = video_dec(video_features)   # (1, T, 3, 64, 64)
        frames = frames.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        frames = (frames * 255).astype(np.uint8)

    # 解码音频
    audio_wave = audio_dec(audio_feature).squeeze().detach().cpu().numpy()
    audio_wave = audio_wave / (np.max(np.abs(audio_wave)) + 1e-8)

    # 保存临时文件
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    temp_video = Path(config.OUTPUT_DIR) / "temp_video.mp4"
    temp_audio = Path(config.OUTPUT_DIR) / "temp_audio.wav"
    output_mp4 = Path(config.OUTPUT_DIR) / "generated.mp4"

    writer = imageio.get_writer(temp_video, fps=config.FPS, format='FFMPEG', codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    sf.write(temp_audio, audio_wave, config.AUDIO_SAMPLE_RATE)

    # 合并音视频
    ffmpeg_exe = None
    # 尝试自动找到 ffmpeg，如果失败则手动指定
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except:
        ffmpeg_exe = r"E:\python\envs\video_gen\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"

    cmd = [
        ffmpeg_exe, '-y', '-i', str(temp_video), '-i', str(temp_audio),
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(output_mp4)
    ]
    subprocess.run(cmd, capture_output=True)

    temp_video.unlink()
    temp_audio.unlink()
    print(f"生成完成: {output_mp4}")

if __name__ == "__main__":
    main()