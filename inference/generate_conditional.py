import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import imageio
import soundfile as sf
import subprocess
from pathlib import Path
from models import VideoEncoder, AudioEncoder, VideoDecoder, AudioDecoder, SimpleDDPMScheduler
from models.text_encoder import SimpleTextEncoder
from models.conditioned_model import ConditionedDiffusionModel
import config

def load_models(device, text_enc_path=None):
    # 加载扩散模型和编码器
    ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "conditional_checkpoint.pth"), map_location=device)
    video_enc = VideoEncoder(output_dim=config.D_MODEL).to(device)
    audio_enc = AudioEncoder(input_length=config.AUDIO_LEN, output_dim=config.D_MODEL).to(device)
    model = ConditionedDiffusionModel(
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        cond_dim=config.TEXT_EMBED_DIM
    ).to(device)
    video_enc.load_state_dict(ckpt['video_enc_state'])
    audio_enc.load_state_dict(ckpt['audio_enc_state'])
    model.load_state_dict(ckpt['model_state'])

    # 文本编码器
    text_enc = SimpleTextEncoder(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=64,
        output_dim=config.TEXT_EMBED_DIM,
        max_len=config.MAX_TEXT_LEN
    ).to(device)
    if text_enc_path is None:
        text_enc.load_state_dict(ckpt['text_enc_state'])

    # 解码器
    dec_ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "decoders.pth"), map_location=device)
    video_dec = VideoDecoder(input_dim=config.D_MODEL).to(device)
    audio_dec = AudioDecoder(input_dim=config.D_MODEL, output_length=config.AUDIO_LEN).to(device)
    video_dec.load_state_dict(dec_ckpt['video_dec_state'])
    audio_dec.load_state_dict(dec_ckpt['audio_dec_state'])

    video_enc.eval(); audio_enc.eval(); model.eval(); video_dec.eval(); audio_dec.eval(); text_enc.eval()
    return video_enc, audio_enc, model, video_dec, audio_dec, text_enc

def generate_with_text(prompt, device='cpu', output_path="outputs/conditional_generated.mp4"):
    video_enc, audio_enc, model, video_dec, audio_dec, text_enc = load_models(device)

    # 编码文本
    with torch.no_grad():
        cond_vec = text_enc.encode(prompt, config.MAX_TEXT_LEN).to(device)   # (1, cond_dim)

    # 采样
    scheduler = SimpleDDPMScheduler(
        num_timesteps=config.NUM_TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    seq_len = config.T_VIDEO + 1
    x = torch.randn(1, seq_len, config.D_MODEL, device=device)

    for t in reversed(range(config.NUM_TIMESTEPS)):
        with torch.no_grad():
            pred_video, _ = model(x, cond_vec)
        beta_t = scheduler.betas[t]
        alpha_bar = scheduler.alpha_bars[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        x = (1 / sqrt_alpha_bar) * (x - (beta_t / torch.sqrt(1 - alpha_bar)) * pred_video)
        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)

    # 解码
    video_features = x[0, :config.T_VIDEO, :].unsqueeze(0)
    audio_feature = x[0, config.T_VIDEO, :].unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        frames = video_dec(video_features).squeeze(0).permute(0,2,3,1).cpu().numpy()
        frames = (frames * 255).astype(np.uint8)
        audio_wave = audio_dec(audio_feature).squeeze().cpu().numpy()
        audio_wave = audio_wave / (np.max(np.abs(audio_wave)) + 1e-8)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_vid = Path("temp_cond_vid.mp4")
    temp_aud = Path("temp_cond_aud.wav")
    writer = imageio.get_writer(temp_vid, fps=config.FPS, format='FFMPEG', codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    sf.write(temp_aud, audio_wave, config.AUDIO_SAMPLE_RATE)

    # 合并
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, '-y', '-i', str(temp_vid), '-i', str(temp_aud),
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_path]
    subprocess.run(cmd, capture_output=True)
    temp_vid.unlink(); temp_aud.unlink()
    print(f"生成完成: {output_path}")

if __name__ == "__main__":
    # 示例：从命令行参数读取提示词
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "a red ball rolling to the right"
    generate_with_text(prompt, device=config.DEVICE)