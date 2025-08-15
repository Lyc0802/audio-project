import numpy as np
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf

# 載入 decoder
decoder = load_model('models/VAE/decoder.h5')

# 產生隨機 latent 向量 (符合標準正態分布)
latent_dim = 64
z_sample = np.random.normal(size=(1, latent_dim))

# 生成 Mel Spectrogram (值域在 [0,1])
generated_mel = decoder.predict(z_sample)[0]

# 將 Mel Spectrogram 還原成音訊
def mel_to_audio(mel_norm, sr=16000, n_fft=1024, hop_length=256):
    import librosa
    mel_db = mel_norm * 80 - 80  # 還原 dB 範圍 [-80,0]
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=60
    )
    return audio

audio = mel_to_audio(generated_mel.squeeze())

# 存成 wav
sf.write('generated_audio.wav', audio, 16000)
print("生成音色已存成 generated_audio.wav，可以聽看看！")
