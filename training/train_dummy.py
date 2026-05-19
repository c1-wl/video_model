import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import SimpleSingleStreamTransformer, VideoEncoder, AudioEncoder
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
    loss_fn = nn.MSELoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS_DUMMY):
        for video, audio in dataloader:
            video = video.to(config.DEVICE)
            audio = audio.to(config.DEVICE)

            video_tokens = video_enc(video)
            audio_tokens = audio_enc(audio)
            x = torch.cat([video_tokens, audio_tokens], dim=1)

            noise = torch.randn_like(x)
            sigma = 0.1
            noisy_x = x + sigma * noise
            pred_video, pred_audio = model(noisy_x)

            loss = loss_fn(pred_video, noise) + loss_fn(pred_audio, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS_DUMMY}, Loss: {loss.item():.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'video_enc_state_dict': video_enc.state_dict(),
        'audio_enc_state_dict': audio_enc.state_dict(),
    }, os.path.join(config.CHECKPOINT_DIR, "checkpoint_last.pth"))
    print("模型已保存")

if __name__ == "__main__":
    main()