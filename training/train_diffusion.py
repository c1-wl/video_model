import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import SimpleSingleStreamTransformer, VideoEncoder, AudioEncoder, SimpleDDPMScheduler
from data_loading.dataset import TinyVideoAudioDataset
import config

def main():
    dataset = TinyVideoAudioDataset(
        config.DATA_DIR,
        video_size=config.VIDEO_SIZE,
        fps=config.FPS,
        audio_sample_rate=config.AUDIO_SAMPLE_RATE,
        duration=config.DURATION
    )
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    video_enc = VideoEncoder(output_dim=config.D_MODEL).to(config.DEVICE)
    audio_enc = AudioEncoder(input_length=config.AUDIO_LEN, output_dim=config.D_MODEL).to(config.DEVICE)
    model = SimpleSingleStreamTransformer(
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(
        list(video_enc.parameters()) + list(audio_enc.parameters()) + list(model.parameters()),
        lr=config.LR
    )
    scheduler = SimpleDDPMScheduler(
        num_timesteps=config.NUM_TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=config.DEVICE
    )
    loss_fn = nn.MSELoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS_DIFFUSION):
        for video, audio in dataloader:
            video = video.to(config.DEVICE)
            audio = audio.to(config.DEVICE)

            video_tokens = video_enc(video)
            audio_tokens = audio_enc(audio)
            x0 = torch.cat([video_tokens, audio_tokens], dim=1)

            t = scheduler.sample_timesteps(config.BATCH_SIZE)
            noise = torch.randn_like(x0)
            x_noisy = scheduler.add_noise(x0, noise, t)

            pred_video, pred_audio = model(x_noisy)
            loss = loss_fn(pred_video, noise) + loss_fn(pred_audio, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS_DIFFUSION}, Loss: {loss.item():.4f}")

    torch.save({
        'model_state': model.state_dict(),
        'video_enc_state': video_enc.state_dict(),
        'audio_enc_state': audio_enc.state_dict(),
        'scheduler_config': {
            'num_timesteps': scheduler.num_timesteps,
            'betas': scheduler.betas.cpu()
        }
    }, os.path.join(config.CHECKPOINT_DIR, "diffusion_checkpoint.pth"))
    print("扩散模型训练完成")

if __name__ == "__main__":
    main()