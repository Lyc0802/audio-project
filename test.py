import tensorflow as tf

# TFRecord 檔案路徑，改成你的路徑
tfrecord_files = ['./downloads/extracted/TAR.down.mage.tens.org_data_nsyn_nsyn-trai.tfrCRSeoUIPAaZd4QoxC56U7MdPFkC4X2j7kSK7ygtdIHo.tar/nsynth-train.tfrecord-00000-of-00512']

raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

# 先試著讀一筆不解析，直接用 tf.train.Example 解出來看欄位
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    print("這筆資料欄位有：")
    for key, feature in example.features.feature.items():
        # 判斷是 int64_list、bytes_list、float_list
        kind = feature.WhichOneof('kind')
        if kind == 'int64_list':
            value = feature.int64_list.value
        elif kind == 'bytes_list':
            value = feature.bytes_list.value
        else:
            value = feature.float_list.value

        print(f"- {key}: 類型={kind}, 長度={len(value)}")
