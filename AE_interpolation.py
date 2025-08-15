import librosa
import librosa.display
import soundfile as sf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

# 假設你已經有：
# encoder 和 autoencoder（包含 decoder）
# data 是你的 Mel spectrogram 資料 (N, 128, 64, 1)
data = np.load('nsynth_train_mels_fixed64.npz')['mels'] 

model_name = 'autoencoder'
encoder= load_model(f'models/{model_name}/encoder.h5', compile=False)
autoencoder= load_model(f'models/{model_name}/autoencoder.h5', compile=False)
# 取兩個不同音色的 Mel
mel1 = data[0:1]  # shape (1,128,64,1)
mel2 = data[1:2]

# 將 Mel 編碼成 latent vector
latent1 = encoder.predict(mel1)  # (1, 16, 8, 128)
latent2 = encoder.predict(mel2)

# 插值函數
def interpolate_latent(l1, l2, alpha):
    return l1 * (1 - alpha) + l2 * alpha

# 產生 5 個插值 latent vector，從 mel1 → mel2
alphas = np.linspace(0, 1, 5)
latents_interp = [interpolate_latent(latent1, latent2, a) for a in alphas]
latents_interp = np.concatenate(latents_interp, axis=0)  # (5, 16, 8, 128)

# 用 Decoder 產生重建 Mel
# Decoder 等於 autoencoder 的輸出層，但我們需要單獨拿出 decoder 模型
# 這裡假設你沒存 decoder，可以用下面程式幫你取出 decoder：
from tensorflow.keras.models import Model
decoder_input = layers.Input(shape=latent1.shape[1:])
x = autoencoder.layers[-5](decoder_input)
x = autoencoder.layers[-4](x)
x = autoencoder.layers[-3](x)
x = autoencoder.layers[-2](x)
decoder_output = autoencoder.layers[-1](x)
decoder = Model(decoder_input, decoder_output)

# 用 decoder 解碼
mels_recon = decoder.predict(latents_interp)  # (5, 128, 64, 1)

# Mel → 音訊函數
def mel_to_audio(mel_norm, sr=16000, n_fft=1024, hop_length=256):
    mel_db = mel_norm.squeeze() * 80 - 80  # 還原 dB
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=60)
    return audio

# 轉成聲音並存檔
for i, mel in enumerate(mels_recon):
    audio = mel_to_audio(mel)
    sf.write(f'interpolated_{i}.wav', audio, 16000)

print("插值生成的音色已儲存成 interpolated_0.wav ~ interpolated_4.wav")
