from abc import ABC, abstractmethod
from data_helper import DataHelper
import numpy as np
import pickle
from keras.models import model_from_json
import tensorflow as tf


class NeuralNetworkBase(ABC):
	"""
	Base class for all neural network models.
	"""
	def __init__(self, model_name=None):
		"""
		Constructor settign up necessary member variables.
		:param model_name: Name of the model used for saving the model
		"""
		self.dh = DataHelper()
		self.columns = ['pool','day_of_week','month','minute_of_day','year'] 
		self.time_steps_back = 5
		self.model = None
		self.model_name = model_name

	@abstractmethod
	def build_model(self):
		"""
		Abstrac function that builds model. Must be implemented by child class.
		In here should be implemented neural network keras model in self.model
		"""
		pass

	def fit_with_training_data(self, epochs=10, validation_split=0.3, use_cpu=True):
		"""
		Trains model for defined number of epochs on training data.
		:param epochs: Number of epochs for fitting.
		:param validation_split: Percentage of training data that will be used for validation
		:param use_cpu: If True it will force to use CPU even if GPU is available
		"""
		x_train, y_train, x_test, y_test = self.dh.generate_feature_vectors_for_cnn(self.columns, self.time_steps_back)
		self.fit(x_train, y_train, epochs)

	def fit(self, x_train, y_train, epochs=10, validation_split=0.3, use_cpu=True):
		"""
		Trains model for defined number of epochs on provided data.
		:param x_train: Numpy array with training data
		:param y_train: Numpy array with ground truth results for training data
		:param epochs: Number of epochs for fitting.
		:param validation_split: Percentage of training data that will be used for validation
		:param use_cpu: If True it will force to use CPU even if GPU is available
		"""
		if self.model is not None:
			if use_cpu:
				with tf.device('/cpu:0'):
					self.model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split)
			else:
				self.model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split)
			self.save_model()
		else:
			print('Cannot fit. Build model first.')

	def predict(self, x):
		"""
		Predicts output for given data.
		:param x: Numpy array with data for prediction
		:return: Prediction output
		"""
		return self.model.predict(x)

	def save_model(self):
		"""
		Saves model and weights with name defined in model_name
		"""
		model_json = self.model.to_json()
		with open(self.model_name+'.json', 'w') as json_file:
		    json_file.write(model_json)

		self.model.save_weights(self.model_name+'_weights.h5')
		print('Model saved to disk with name:' + self.model_name)

	def load_model(self):
		"""
		Loads model and weights with name defined in model_name
		"""
		json_file = open(self.model_name+'.json', 'r')
		model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(model_json)
		self.model.load_weights(self.model_name+'_weights.h5')
		print('Model %s loaded from disk'%(self.model_name))

	def setup(self, columns, time_steps_back):
		"""
		Trains model for defined number of epochs on provided data.
		:param columns: Colmuns to keep in data
		:param time_steps_back: Number of time steps used for prediction
		"""
		self.columns = columns
		self.time_steps_back = time_steps_back
		self.build_model()
