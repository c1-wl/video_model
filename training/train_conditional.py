import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import VideoEncoder, AudioEncoder
from models.text_encoder import SimpleTextEncoder   # 或 CLIPTextEncoder
from models.conditioned_model import ConditionedDiffusionModel
from models.ddpm_scheduler import SimpleDDPMScheduler
from data_loading.dataset_conditional import ConditionalVideoAudioDataset
import config

def main():
    device = config.DEVICE

    # 初始化文本编码器（简单版本）
    text_enc = SimpleTextEncoder(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=64,
        output_dim=config.TEXT_EMBED_DIM,
        max_len=config.MAX_TEXT_LEN
    ).to(device)

    # 数据集
    dataset = ConditionalVideoAudioDataset(
        data_dir=config.DATA_DIR,
        text_encoder=text_enc,   # 注意：text_encoder需要实现encode方法
        video_size=config.VIDEO_SIZE,
        fps=config.FPS,
        audio_sample_rate=config.AUDIO_SAMPLE_RATE,
        duration=config.DURATION,
        max_text_len=config.MAX_TEXT_LEN
    )
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 编码器
    video_enc = VideoEncoder(output_dim=config.D_MODEL).to(device)
    audio_enc = AudioEncoder(input_length=config.AUDIO_LEN, output_dim=config.D_MODEL).to(device)

    # 条件扩散模型
    model = ConditionedDiffusionModel(
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN,
        cond_dim=config.TEXT_EMBED_DIM
    ).to(device)

    optimizer = torch.optim.Adam(
        list(video_enc.parameters()) + list(audio_enc.parameters()) + list(model.parameters()),
        lr=config.COND_LR
    )
    scheduler = SimpleDDPMScheduler(
        num_timesteps=config.NUM_TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    loss_fn = nn.MSELoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS_COND):
        for video, audio, cond_vec in dataloader:
            video = video.to(device)
            audio = audio.to(device)
            cond_vec = cond_vec.to(device).squeeze(1)   # (B, cond_dim)

            # 编码音视频为潜空间特征
            video_tokens = video_enc(video)
            audio_tokens = audio_enc(audio)
            x0 = torch.cat([video_tokens, audio_tokens], dim=1)   # (B, L, D)

            t = scheduler.sample_timesteps(config.BATCH_SIZE)
            noise = torch.randn_like(x0)
            x_noisy = scheduler.add_noise(x0, noise, t)

            # 条件预测噪声
            pred_video, pred_audio = model(x_noisy, cond_vec)
            loss = loss_fn(pred_video, noise) + loss_fn(pred_audio, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS_COND}, Loss: {loss.item():.4f}")

    # 保存模型
    torch.save({
        'model_state': model.state_dict(),
        'video_enc_state': video_enc.state_dict(),
        'audio_enc_state': audio_enc.state_dict(),
        'text_enc_state': text_enc.state_dict(),
        'scheduler_config': {'betas': scheduler.betas.cpu()}
    }, os.path.join(config.CHECKPOINT_DIR, "conditional_checkpoint.pth"))
    print("条件扩散模型训练完成")

if __name__ == "__main__":
    main()