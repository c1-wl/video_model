import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import VideoEncoder, AudioEncoder, VideoDecoder, AudioDecoder
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

    # 加载训练好的扩散模型中的编码器
    diff_ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "diffusion_checkpoint.pth"), map_location=config.DEVICE)
    video_enc = VideoEncoder(output_dim=config.D_MODEL).to(config.DEVICE)
    audio_enc = AudioEncoder(input_length=config.AUDIO_LEN, output_dim=config.D_MODEL).to(config.DEVICE)
    video_enc.load_state_dict(diff_ckpt['video_enc_state'])
    audio_enc.load_state_dict(diff_ckpt['audio_enc_state'])
    video_enc.eval()
    audio_enc.eval()

    video_dec = VideoDecoder(input_dim=config.D_MODEL).to(config.DEVICE)
    audio_dec = AudioDecoder(input_dim=config.D_MODEL, output_length=config.AUDIO_LEN).to(config.DEVICE)

    video_optim = torch.optim.Adam(video_dec.parameters(), lr=config.LR)
    audio_optim = torch.optim.Adam(audio_dec.parameters(), lr=config.LR)
    loss_fn = nn.MSELoss()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS_DECODER):
        for video, audio in dataloader:
            video = video.to(config.DEVICE)
            audio = audio.to(config.DEVICE)

            with torch.no_grad():
                video_tokens = video_enc(video)
                audio_tokens = audio_enc(audio)

            pred_frames = video_dec(video_tokens)
            v_loss = loss_fn(pred_frames, video)

            pred_audio_wave = audio_dec(audio_tokens).squeeze(1)
            a_loss = loss_fn(pred_audio_wave, audio)

            video_optim.zero_grad()
            v_loss.backward()
            video_optim.step()

            audio_optim.zero_grad()
            a_loss.backward()
            audio_optim.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS_DECODER}, V_loss: {v_loss.item():.4f}, A_loss: {a_loss.item():.4f}")

    torch.save({
        'video_dec_state': video_dec.state_dict(),
        'audio_dec_state': audio_dec.state_dict(),
    }, os.path.join(config.CHECKPOINT_DIR, "decoders.pth"))
    print("解码器训练完成")

if __name__ == "__main__":
    main()