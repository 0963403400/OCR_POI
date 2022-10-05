
class Hparams:
	def __init__(self):
		### data and save path
		self.train_record_path = '/home/tuan291100/Desktop/OCR_POI/data2/data.train'
		self.num_train_sample = 4664
		self.valid_record_path = '/home/tuan291100/Desktop/OCR_POI/data2/data.valid'
		self.charset_path = '/home/tuan291100/Desktop/OCR_POI/charset_size=64.txt'
		self.num_valid_sample = 535
		# self.save_path = '/data1/users/tuanb/OCR_DOI/training_checkpoints/hparams_1'
		self.save_path = '/home/tuan291100/Desktop/OCR_POI/checkpoints'
		self.save_best = True
		self.max_to_keep = 1000

		### input params
		self.image_shape = (64, 320, 3)
		self.nul_code = 63
		self.charset_size = 64
		self.max_char_length = 6

		### conv_tower params
		# base model from tf.keras.application, or custom instance of tf.keras.Model
		# check for new models from https://www.tensorflow.org/api_docs/python/tf/keras/applications
		# check for newest model from tf-nightly version
		self.base_model_name = 'EfficientNetB1'
		# last convolution layer from base model which extract features from
		# inception v3: mixed2 (mixed_5d in tf.slim inceptionv3)
		# inception resnet v2: (mixed_6a in tf.slim inception_resnet_v2)
		self.end_point = 'block6e_add'
		# endcode cordinate feature to conv_feature
		self.use_encode_cordinate = True

		### RNN tower params
		self.rnn_cell = 'bilstm'
		self.rnn_units = 256
		self.dense_units = 256
		self.weight_decay = 0.00004

		### attention params
		# self.model_size = 256
		# self.num_heads = 8
		# h

		### training params
		self.batch_size = 64
		self.max_epochs = 10
		self.lr = 0.0001

hparams = Hparams()
