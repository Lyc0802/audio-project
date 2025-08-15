# latent_arithmetic.py
import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# ========= 你的訓練設定（務必與訓練一致）=========
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
FIXED_FRAMES = 64       # 128 x 64 x 1
LATENT_DIM = 64

ENCODER_PATH = "models/VAE/encoder.h5"
DECODER_PATH = "models/VAE/decoder.h5"

# ========= 自訂 sampling（用來成功載入 encoder.h5）=========
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# ========= 前/後處理 =========
def wav_to_mel_db01(wav_path):
    """讀 wav → 16k 重採樣 → mel(power) → dB → 正規化到 [0,1] → 固定 64 幀 → (128,64,1)"""
    y, sr = sf.read(wav_path)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    # 正規化到 [-1,1]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db01 = (mel_db + 80.0) / 80.0  # [-80,0] -> [0,1]

    # 固定幀長
    if mel_db01.shape[1] < FIXED_FRAMES:
        pad_w = FIXED_FRAMES - mel_db01.shape[1]
        mel_db01 = np.pad(mel_db01, ((0,0),(0,pad_w)), mode="constant")
    else:
        mel_db01 = mel_db01[:, :FIXED_FRAMES]

    return mel_db01[..., np.newaxis].astype(np.float32)  # (128,64,1)

def mel_db01_to_wav(mel_db01, out_wav):
    """(128,64,1) 的 mel dB[0,1] → dB → power → Griffin-Lim → 存 wav"""
    mel_db01 = np.squeeze(mel_db01)      # (128,64)
    mel_db = mel_db01 * 80.0 - 80.0      # [0,1] -> [-80,0]
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        M=mel_power,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=60
    )
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(out_wav, audio, SR)
    return out_wav

def load_mel_npy(npy_path):
    """載入單檔 .npy（應為 (128,64,1) 或 (1,128,64,1)）"""
    arr = np.load(npy_path)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    return arr.astype(np.float32)

def load_mel_from_npz(npz_path, idx):
    """從 mels_fixed64.npz 以索引取一筆 (1,128,64,1)"""
    z = np.load(npz_path, allow_pickle=True)
    mels = z["mels"]
    mel = mels[idx]
    return mel[np.newaxis, ...].astype(np.float32)

# ========= 模型推論工具 =========
def load_models(encoder_path=ENCODER_PATH, decoder_path=DECODER_PATH):
    encoder = load_model(encoder_path, custom_objects={"sampling": sampling}, compile=False)
    decoder = load_model(decoder_path, compile=False)
    return encoder, decoder

def encode_mels(encoder, mels_bhwc, use_z_mean=True):
    """mels: (B,128,64,1) -> z 向量"""
    z_mean, z_log_var, z = encoder.predict(mels_bhwc, verbose=0)
    return z_mean if use_z_mean else z

def decode_mels(decoder, z):
    """z: (B, LATENT_DIM) -> mels (B,128,64,1)"""
    mels = decoder.predict(z, verbose=0)
    return np.clip(mels, 0.0, 1.0)

def latent_arithmetic(zA, zB, zC, alpha=1.0):
    """zA - zB + zC 的一般化：zA + alpha*(zC - zB)"""
    return zA + alpha * (zC - zB)

# ========= CLI 參數解析 =========
def build_parser():
    p = argparse.ArgumentParser(description="VAE Latent Arithmetic: zA - zB + zC -> WAV")
    # 模型路徑
    p.add_argument("--encoder", default=ENCODER_PATH)
    p.add_argument("--decoder", default=DECODER_PATH)

    # 來源：可用 wav / npy / npz+idx 任選其一（A/B/C 各自選一種）
    p.add_argument("--a_wav")
    p.add_argument("--b_wav")
    p.add_argument("--c_wav")

    p.add_argument("--a_npy")
    p.add_argument("--b_npy")
    p.add_argument("--c_npy")

    p.add_argument("--npz")           # 若要用 npz，提供這個路徑
    p.add_argument("--a_idx", type=int)
    p.add_argument("--b_idx", type=int)
    p.add_argument("--c_idx", type=int)

    # 其它
    p.add_argument("--alpha", type=float, default=1.0, help="scale for (zC - zB)")
    p.add_argument("--use_z_mean", action="store_true", default=True, help="use z_mean (default)")
    p.add_argument("--use_z", dest="use_z_mean", action="store_false", help="use sampled z instead of z_mean")
    p.add_argument("--out_wav", default="latent_results/A_minus_B_plus_C.wav")
    return p

def get_mel_for_role(args, role_prefix):
    """
    依 args 中的指定載入 A/B/C 的 mel：(1,128,64,1)
    優先順序：*_wav > *_npy > (npz & *_idx)
    """
    wav = getattr(args, f"{role_prefix}_wav")
    npy = getattr(args, f"{role_prefix}_npy")
    idx = getattr(args, f"{role_prefix}_idx")

    if wav is not None:
        mel = wav_to_mel_db01(wav)[np.newaxis, ...]
        return mel
    if npy is not None:
        return load_mel_npy(npy)
    if args.npz is not None and idx is not None:
        return load_mel_from_npz(args.npz, idx)
    raise ValueError(f"請為 {role_prefix.upper()} 提供來源之一：--{role_prefix}_wav 或 --{role_prefix}_npy 或 (--npz 與 --{role_prefix}_idx)")

if __name__ == "__main__":
    os.makedirs("latent_results", exist_ok=True)

    # ========= 載入模型 =========
    encoder, decoder = load_models()

    # ========= A) 直接用三個 wav 做 zA - zB + zC =========
    wav_A = "nsynth_sample_wav/guitar/guitar_0.wav"   # 目標音色 A
    wav_B = "nsynth_sample_wav/keyboard/keyboard_0.wav"     # 參考音色 B
    wav_C = "nsynth_sample_wav/flute/flute_0.wav"   # 要把 C「變得像 A 去除 B 的特徵」

    melA = wav_to_mel_db01(wav_A)[np.newaxis, ...]    # (1,128,64,1)
    melB = wav_to_mel_db01(wav_B)[np.newaxis, ...]
    melC = wav_to_mel_db01(wav_C)[np.newaxis, ...]

    zA = encode_mels(encoder, melA)
    zB = encode_mels(encoder, melB)
    zC = encode_mels(encoder, melC)

    z_new = latent_arithmetic(zA, zB, zC)             # zA - zB + zC
    mel_new = decode_mels(decoder, z_new)[0]          # (128,64,1)
    out_wav_A = "latent_results/A_minus_B_plus_C.wav"
    mel_db01_to_wav(mel_new, out_wav_A)
    print("Saved:", out_wav_A)

    # ========= B) 用資料夾平均向量 =========
    # 需要先有 mels_fixed64.npz 與 files 欄位
    npz_path = "nsynth_sample_mels/mels_fixed64.npz"

    def load_from_npz(npz_path, file_substring):
        data = np.load(npz_path, allow_pickle=True)
        mels = data["mels"]
        files = data.get("files", None)
        if files is None:
            raise ValueError("npz 需要有 files 欄位")
        subset = [m for m,f in zip(mels, files) if file_substring in f]
        return np.array(subset), [f for f in files if file_substring in f]

    def average_latent_over_subset(mels_subset):
        if len(mels_subset) == 0:
            return None
        z = encode_mels(encoder, mels_subset)
        return np.mean(z, axis=0, keepdims=True)
    
    instrument_A = "guitar"
    instrument_B = "keyboard"

    A_mels, _ = load_from_npz(npz_path, file_substring="guitar")
    B_mels, _ = load_from_npz(npz_path, file_substring="keyboard")

    if len(A_mels) > 0 and len(B_mels) > 0:
        z_violin_avg = average_latent_over_subset(A_mels)
        z_piano_avg = average_latent_over_subset(B_mels)
        # 用前面 C 的單一 wav
        z_mix = latent_arithmetic(z_violin_avg, z_piano_avg, zC)
        mel_mix = decode_mels(decoder, z_mix)[0]
        out_wav_mix = f"latent_results/{instrument_A}_avg_minus_{instrument_B}_avg_plus_C.wav"
        mel_db01_to_wav(mel_mix, out_wav_mix)
        print("Saved:", out_wav_mix)
    else:
        print(f"找不到 {instrument_A}/{instrument_B} 相關樣本，請確認 npz 的 files 欄位關鍵字。")
