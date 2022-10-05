import tensorflow as tf
import argparse
import cv2
import os


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # print("_bytes_feature", type(value))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    # print('_int64_list_feature is OK')
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(image_encoded, image_format, image_width, image_orig_width, \
                        image_class, unpadded_class, image_text):
    feature = {
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(image_format),
        'image/width': _int64_feature(image_width),
        'image/orig_width': _int64_feature(image_orig_width),
        'image/class': _int64_list_feature(image_class),
        'image/unpadded_class': _int64_list_feature(unpadded_class),
        'image/text': _bytes_feature(image_text)
    }
  
    # Create a Features message using tf.train.Example.
  
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_example_(image_encoded, image_class, image_text):
    # feature = {
    #     'image/encoded': _bytes_feature(image_encoded),
    #     'image/class': _int64_list_feature(image_class),
    #     'image/text': _bytes_feature(image_text)
    # }
    # print('feature: ', feature)
    # print('image/encoded: ',_bytes_feature(image_encoded))
    # print('image/class: ', type(feature.get('image/class')))
    # print('image/text: ', type(feature.get('image/text')))
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_encoded),
        'image/class': _int64_list_feature(image_class),
        'image/text': _bytes_feature(image_text)
    }))
    example_proto = example_proto.SerializeToString()
    # print(example_proto)
    return example_proto
def main(args):
    null_id = 63 #doi, doe  
    max_seqlen = 6 #doi, doe
    #null_id = 208 #resident
    #max_seqlen = 16  #resident
    charset_path = args.charset_path
    save_path = args.out_path
    # read image filenames
    filenames = []
    texts = []
    
    with open(os.path.join(args.pad_path, "label.txt"), "r") as f:
        for row in f:
            split_row = row[:-1].split("\t")
            filenames.append(split_row[0])
            texts.append(split_row[1])
            # print('filename\t label: ',split_row[0], split_row[1])
    # read character set
    charset = {}
    with open(charset_path, "r") as f:
        for row in f:
            val, key = row[:-1].split("\t")
            charset[key] = int(val)
            # print('val\tkey:', val, key)

    cnt = 0
    error_files = []
    with tf.io.TFRecordWriter(save_path) as writer:
        for i, (filename, text) in enumerate(zip(filenames, texts)):
            # print(i, filename, "\t", text)
            # text = text.upper()
            if len(text) > max_seqlen:
                continue
            ### prepare all feature values
            # image/encoded
            try:
                # print(">>>>>>>>>>>Start create record file:")
                with open(os.path.join(args.pad_path, filename), "rb") as f:
                    # print(os.path.join(args.pad_path, filename))
                    image_encoded = f.read()
                cnt += 1
                print('image_encoded:', len(image_encoded))
                '''
                # image/format
           
                image_format = "png".encode()
                # image/width
               
                image_width = 320 #doi, resident
                # image/orig_width
                h, w, _ = cv2.imread(os.path.join(args.unpad_path, filename)).shape
                
                image_orig_width = w
                '''
                # image/class
                image_class = []
                for char in text:
                    image_class.append(charset[char])
                while len(image_class) < max_seqlen:
                    image_class.append(null_id)
                print('image_class: ', image_class)
                '''
                # image/unpadded_class
                unpadded_class = []
                for char in text:
                    unpadded_class.append(charset[char])
                '''
                # image/text
                image_text = text.encode()
                # write to TFRecordFile
               
                
                # print('unpadded_class: ', unpadded_class)
                # print('image_text: ', type(image_text))
                # example = serialize_example(image_encoded, image_format, image_width, image_orig_width, \
                #                             image_class, unpadded_class, image_text)
                example = serialize_example_(image_encoded, image_class, image_text)
                # print(len(example))
                writer.write(example)
            except:
                print('error')
            
    print()
    print(cnt)
    for file in error_files:
        print(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", help="Generate train data or valid data", default="train")
    parser.add_argument("--pad_path", help = "directory contain images", default = "/content/drive/MyDrive/OCR_MTA/OCR_POI/data/pad_train")
    parser.add_argument("--unpad_path", help = "directory contain images", default = "/content/drive/MyDrive/OCR_MTA/OCR_POI/data/unpad_train")
    parser.add_argument("--charset_path", help = "character set file path", default = "/home/tuan291100/Desktop/OCR_POI/charset_size=64.txt")
    parser.add_argument("--out_path", help = "output path", default = "/content/drive/MyDrive/OCR_MTA/OCR_POI/records/data_v2.train")
    args = parser.parse_args()
    main(args)

"""
    python gen_record.py --pad_path="/home/tuan291100/Desktop/OCR_POI/datasets2/pad_train" --unpad_path="/home/tuan291100/Desktop/OCR_POI/datasets2/unpad_train" --out_path="/home/tuan291100/Desktop/OCR_POI/datasets2/data.train"
    python gen_record.py --pad_path="/home/tuan291100/Desktop/OCR_POI/datasets2/pad_valid" --unpad_path="/home/tuan291100/Desktop/OCR_POI/datasets2/unpad_valid" --out_path="/home/tuan291100/Desktop/OCR_POI/datasets2/data.valid"
"""
