import tensorflow as tf
import numpy as np
import librosa

n_dataset = 64

base_path = './downloads/extracted/TAR.down.mage.tens.org_data_nsyn_nsyn-trai.tfrCRSeoUIPAaZd4QoxC56U7MdPFkC4X2j7kSK7ygtdIHo.tar/'
tfrecord_files = [f"{base_path}nsynth-train.tfrecord-{str(i).zfill(5)}-of-00512" for i in range(n_dataset)]


feature_description = {
    'audio': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
}

def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    audio = parsed['audio']
    return audio

def audio_to_mel(audio, sr=16000, n_mels=128, hop_length=256, n_fft=1024, fixed_frames=64):
    y = audio / np.max(np.abs(audio))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80) / 80

    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :fixed_frames]

    return mel_db[..., np.newaxis]

all_mels = []

for tfrecord_path in tfrecord_files:
    print(f"Processing {tfrecord_path} ...")
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    for audio_tensor in parsed_dataset:
        audio_np = audio_tensor.numpy()
        mel = audio_to_mel(audio_np)
        all_mels.append(mel)

all_mels_array = np.array(all_mels)
print("All mel spectrograms shape:", all_mels_array.shape)

np.savez_compressed(f'nsynth_train_mels_fixed64_{n_dataset}.npz', mels=all_mels_array)
print(f"Saved to nsynth_train_mels_fixed64_{n_dataset}.npz")
