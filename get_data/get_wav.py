import tensorflow as tf
import numpy as np
import soundfile as sf
import os, random

# 這裡換成你自己的 TFRecord 檔案路徑
tfrecord_files = [
    './downloads/extracted/TAR.down.mage.tens.org_data_nsyn_nsyn-test.tfrZd8NCtpUDKGYOEmgUcvsTQ2cUZALasuKTAZJJWBoCnE.tar/nsynth-test.tfrecord-00000-of-00008'
]

raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

# 收集每個 family 對應的音訊
family_to_examples = {}

print("正在掃描 TFRecord ...")
for idx, raw_record in enumerate(raw_dataset):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    # 檢查有沒有 audio 欄位
    if 'audio' not in example.features.feature:
        continue
    feat = example.features.feature['audio']

    # 判斷 audio 是 bytes_list 還是 float_list
    if feat.bytes_list.value:  # PCM int16 bytes
        audio_bytes = feat.bytes_list.value[0]
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif feat.float_list.value:  # float waveform
        audio_np = np.array(feat.float_list.value, dtype=np.float32)
    else:
        continue  # 空的直接跳過

    # 讀取 instrument family
    fam = example.features.feature['instrument_family_str'].bytes_list.value[0].decode('utf-8')
    family_to_examples.setdefault(fam, []).append(audio_np)

    if idx % 500 == 0:
        print(f"已讀取 {idx} 筆...")

print("讀取完成，各 family 數量：")
for fam, samples in family_to_examples.items():
    print(f"{fam}: {len(samples)}")

# 建立輸出資料夾
out_dir = "nsynth_sample_wav"
os.makedirs(out_dir, exist_ok=True)

# 每個 family 隨機抽 5 個
for fam, samples in family_to_examples.items():
    fam_dir = os.path.join(out_dir, fam)
    os.makedirs(fam_dir, exist_ok=True)

    pick = random.sample(samples, min(5, len(samples)))
    for i, audio_np in enumerate(pick):
        fname = os.path.join(fam_dir, f"{fam}_{i}.wav")
        sf.write(fname, audio_np, 16000)

print(f"已完成！檔案存放於 {out_dir}/")
