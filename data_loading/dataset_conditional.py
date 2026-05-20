import torch
import cv2
import numpy as np
import subprocess
from pathlib import Path

class ConditionalVideoAudioDataset(torch.utils.data.Dataset):
    """读取视频、音频和对应的文本描述"""
    def __init__(self, data_dir, text_encoder, video_size=(64,64), fps=8,
                 audio_sample_rate=16000, duration=1.0, max_text_len=32, ffmpeg_path=None):
        self.data_dir = Path(data_dir)
        self.video_size = video_size
        self.fps = fps
        self.audio_sr = audio_sample_rate
        self.duration = duration
        self.max_text_len = max_text_len
        self.text_encoder = text_encoder   # 外部传入，用于将文本转为向量

        self.video_paths = list(self.data_dir.glob("*.mp4")) + list(self.data_dir.glob("*.avi"))
        if not self.video_paths:
            raise FileNotFoundError(f"No video files found in {data_dir}")

        # 检查每个视频是否有对应的 .txt 文件
        self.txt_paths = []
        for vp in self.video_paths:
            txt_path = vp.with_suffix('.txt')
            if not txt_path.exists():
                raise FileNotFoundError(f"Missing text file: {txt_path}")
            self.txt_paths.append(txt_path)

        # ffmpeg 路径
        if ffmpeg_path is None:
            import imageio_ffmpeg
            self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        else:
            self.ffmpeg_exe = ffmpeg_path

    def __len__(self):
        return len(self.video_paths)

    def _read_text(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        # 使用 text_encoder 转换为向量 (1, text_embed_dim)
        text_vec = self.text_encoder.encode(text, self.max_text_len)
        return text_vec

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # ---------- 读取视频帧 ----------
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        target_frames = int(self.duration * self.fps)
        while len(frames) < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.video_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        while len(frames) < target_frames:
            frames.append(frames[-1])
        video = torch.from_numpy(np.array(frames)).float() / 255.0
        video = video.permute(0, 3, 1, 2)   # (T, C, H, W)

        # ---------- 读取音频 ----------
        try:
            cmd = [
                self.ffmpeg_exe, '-i', str(video_path),
                '-f', 'f32le', '-acodec', 'pcm_f32le',
                '-ar', str(self.audio_sr), '-ac', '1',
                '-t', str(self.duration), '-'
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            audio_bytes = proc.stdout.read()
            proc.terminate()
            audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()
            expected_len = int(self.duration * self.audio_sr)
            if len(audio) < expected_len:
                audio = np.pad(audio, (0, expected_len - len(audio)))
            else:
                audio = audio[:expected_len]
        except Exception as e:
            print(f"Warning: failed to read audio from {video_path}: {e}, using zeros")
            audio = np.zeros(int(self.duration * self.audio_sr), dtype=np.float32)
        audio = torch.from_numpy(audio).float()

        # ---------- 读取文本条件 ----------
        text_vec = self._read_text(self.txt_paths[idx])   # (1, text_embed_dim)

        return video, audio, text_vec