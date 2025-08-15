import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 載入 Mel 資料
data = np.load('nsynth_train_mels_fixed64.npz')['mels']  # shape: (N, 128, 64, 1)
print("Loaded data shape:", data.shape)
data = np.clip(data, 0.0, 1.0).astype(np.float32)

# 2. 建立 Autoencoder
input_shape = (128, 64, 1)

# Encoder
encoder_input = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
latent = layers.MaxPooling2D((2, 2), padding='same')(x)  # latent shape (16, 8, 128)

encoder = models.Model(encoder_input, latent, name="encoder")

# Decoder
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(latent)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(encoder_input, decoder_output, name="autoencoder")

# 3. 編譯模型
autoencoder.compile(optimizer='adam', loss='mse')

# 4. 訓練
autoencoder.fit(
    data, data,
    epochs=20,
    batch_size=64,
    shuffle=True,
    validation_split=0.1
)

# 5. 保存模型
autoencoder.save('models/autoencoder/autoencoder.h5')
encoder.save('models/autoencoder/encoder.h5')

# 6. 拆解並建立 Decoder 模型
latent_shape = encoder.output_shape[1:]  # (16, 8, 128)
decoder_input = layers.Input(shape=latent_shape)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_input)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = models.Model(decoder_input, decoder_output, name='decoder')

# 7. 保存 Decoder 模型
decoder.save('models/autoencoder/decoder.h5')

print("Saved autoencoder, encoder, and decoder models.")
