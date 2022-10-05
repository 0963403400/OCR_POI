import tensorflow as tf
import pdb
# import tensorflow as tf
import argparse
import cv2
import os

def load_all_image (record_file_path, save_dir):
      zero = tf.zeros([10], dtype=tf.int64)
      record_iterator = tf.data.TFRecordDataset(record_file_path)
      keys_to_features = {
            'image/encoded': tf.io.RaggedFeature(dtype=tf.string),
            # 'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
            # 'image/width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            # 'image/orig_width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/class':tf.io.RaggedFeature(dtype=tf.int64),
            # 'image/unpadded_class':tf.io.VarLenFeature(tf.int64),
            'image/text': tf.io.RaggedFeature(dtype=tf.string),
            
        }
      print('record_iterator:',record_iterator)
      for string_record in record_iterator:
        example = tf.train.Example()
        res = tf.io.parse_single_example(example, keys_to_features)
        image = tf.cast(tf.io.decode_jpeg(res['image/encoded'], 3), tf.float32)/255.0
        label = tf.cast(res['image/class'], tf.float32)
        example.ParseFromString(string_record)
        image = example.features.feature["encoded"].bytes_list.value[0]
        print("image: ", image)
        print("label: ", label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_file", default = "/content/drive/MyDrive/OCR_MTA/OCR_POI/records/data.valid")
    parser.add_argument("--save_file", default = "/content/drive/MyDrive/OCR_MTA/OCR_POI/data/valid")
    args = parser.parse_args()
    load_all_image(args.record_file, args.save_file)
    print("Load images successfully!")


