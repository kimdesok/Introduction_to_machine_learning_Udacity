import tensorflow as tf
# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class model_data:
	def __init__(self, data_name):
		dataset, metadata = tfds.load(data_name, as_supervised=True, with_info=True)
		self.train_dataset, self.test_dataset = dataset['train'], dataset['test']
		
	def train_data():
		return self.train_dataset
	def test_data():
		return self.test_dataset
	def class_names():
		return self.metadata.features['label'].names
	def train_size():
		return self.metadata.splits['train'].num_examples
	def test_size():
		return self.metadata.splits['test'].num_examples
	def _normalize(images, labels):
		images = tf.cast(images, tf.float32)
		images = images / 125.5 - 1.0 
		labels = labels
		return images, labels
	def normalize(dataset):
		return dataset.map(_normalize)
		
		