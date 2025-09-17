import os
import numpy as np
import librosa
import soundfile as sf

def audio_to_mel(audio, sr=16000, n_mels=128, hop_length=256, n_fft=1024, fixed_frames=64):
    # 正規化到 [-1, 1]
    if np.max(np.abs(audio)) > 0:
        y = audio / np.max(np.abs(audio))
    else:
        y = audio
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80) / 80  # 壓縮到 [0, 1]

    # 固定 frame 長度
    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :fixed_frames]

    return mel_db[..., np.newaxis]

def process_wav_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_mels = []
    file_paths = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_path = os.path.join(root, file)
                print(f"處理 {wav_path} ...")

                audio, sr = sf.read(wav_path)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000

                mel = audio_to_mel(audio, sr=sr)
                all_mels.append(mel)
                file_paths.append(wav_path)

                # 輸出單獨 .npy
                rel_path = os.path.relpath(wav_path, input_dir)  # 相對路徑
                npy_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".npy")
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, mel)

    # 儲存所有 mels 和對應檔案清單
    all_mels_array = np.array(all_mels)
    np.savez_compressed(os.path.join(output_dir, 'mels_fixed64.npz'), mels=all_mels_array, files=file_paths)
    print(f"已完成！共 {len(all_mels)} 筆")
    print(f"集中檔：{os.path.join(output_dir, 'mels_fixed64.npz')}")
    print(f"個別檔已存於：{output_dir}")

if __name__ == "__main__":
    # 這裡改成你的資料夾路徑
    input_dir = "nsynth_sample_wav"      # WAV 資料夾
    output_dir = "nsynth_sample_mels"    # 輸出 Mel 資料夾
    process_wav_folder(input_dir, output_dir)
