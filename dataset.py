import tensorflow as tf
import math 
import pdb
import os
import cv2
import numpy as np
class Dataset(object):
    def __init__(self, hparams, record_path):
        self.hparams = hparams
        self.record_path = record_path
        self.batch_size = 1
        self.max_char_length = 6
        zero = tf.zeros([1], dtype=tf.int64)
        # self.keys_to_features = {
        #     'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        #     # 'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
        #     # 'image/width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
        #     # 'image/orig_width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
        #     'image/class':tf.io.FixedLenFeature([self.max_char_length], tf.int64, default_value=zero),
        #     # 'image/unpadded_class':tf.io.VarLenFeature(tf.int64),
        #     'image/text': tf.io.FixedLenFeature([1], tf.string, default_value=''),
        # }
        self.keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            # 'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
            # 'image/width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            # 'image/orig_width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/class': tf.io.FixedLenFeature([6], tf.int64),
            # 'image/unpadded_class':tf.io.VarLenFeature(tf.int64),
            'image/text': tf.io.FixedLenFeature([], tf.string),
            
        }

    def parse_tfrecord(self, example):
       
        res = tf.io.parse_single_example(example, self.keys_to_features)
        # print('xxxxxxx', res['image/encoded'])
        # print("yyyyyyyyy", res['image/class'])
        # pdb.set_trace()
        # tf.config.run_functions_eagerly(True)
        image = tf.cast(tf.io.decode_png(res['image/encoded'], channels = 0), tf.float32)
        
        # print(type(gray_scale))
        # gray = gray_scale.eval(session=tf.compat.v1.Session())
        # image  = np.dstack((gray, gray, gray))
        # image = tf.constant(image)
        # import pdb;pdb.set_trace();


        label = tf.cast(res['image/class'], tf.int64)
        # orig_width = tf.cast(res['image/orig_width'], tf.int64)
        text = tf.cast(res['image/text'], tf.string)
        
        return image, label, text
        
        # return res

    def load_tfrecord(self, repeat=None):
        dataset = tf.data.TFRecordDataset([self.record_path])
        for raw_record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            print(example)
        dataset = dataset.map(self.parse_tfrecord)
        for parsed_record in dataset.take(10):
          print(repr(parsed_record))
        #   import pdb;pdb.set_trace();
        print("**************************")
        print(dataset)
        #dataset.shuffle(buffer_size=self.hparams.num_train_sample, reshuffle_each_iteration=True)
        self.dataset = dataset.batch(self.batch_size)
        # print('list(self.dataset.as_numpy_iterator()):', list(self.dataset.as_numpy_iterator()))
        #self.dataset = dataset.cache()
        self.iterator = iter(dataset)

    def next_batch(self):
        return self.iterator.get_next()



        
def save_image (train_record_path, save_path):
    train_dataset = Dataset(train_record_path)
    train_dataset.load_tfrecord()
    file_names = []
    labels = []
    widths = []
    image_name = "_image.png"
    count = 0
    # pdb.set_trace()
    for batch, (batch_input, batch_target, batch_text ) in enumerate(train_dataset.dataset):
        image = batch_input[0].numpy()
        label = batch_target[0].numpy()
        # orig_width = batch_orig_width [0].numpy() [0]
        text = str (batch_text[0].numpy()[0] , 'utf-8')
        # print("**************************")
        # print(image)
        # print(label)
        # print(text)
        count += 1
        img_name = str(count) + image_name
        file_name_out = os.path.join (save_path, img_name)
        file_names.append (img_name)
        labels.append (text)
        # widths.append (orig_width)
        cv2.imwrite (file_name_out, image)

       
    with open(os.path.join(save_path, 'label.txt'), 'w') as f:
        for file_name, label, width in zip (file_names, labels, widths):
            f.write(file_name + '\t' + label +  '\t' + str (width)+'\n')

def list_record_features(tfrecords_path):
    # Dict of extracted feature information
    features = {}
    # Iterate records
    for rec in tf.data.TFRecordDataset([str(tfrecords_path)]):
        # Get record bytes
        example_bytes = rec.numpy()
        # Parse example protobuf message
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        # Iterate example features
        for key, value in example.features.feature.items():
            # Kind of data in the feature
            kind = value.WhichOneof('kind')
            # Size of data in the feature
            size = len(getattr(value, kind).value)
            # Check if feature was seen before
            if key in features:
                # Check if values match, use None otherwise
                kind2, size2 = features[key]
                if kind != kind2:
                    kind = None
                if size != size2:
                    size = None
            # Save feature data
            features[key] = (kind, size)
    return features









