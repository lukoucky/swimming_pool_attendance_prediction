from abc import ABC, abstractmethod
from data_helper import DataHelper
import numpy as np
import pickle
from keras.models import model_from_json
import tensorflow as tf
tf.random.set_seed(17)


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
		self.fit_history = {'val_loss':list(), 'loss':list(), 'mse':list()}
		self.time_steps_back = 5
		self.columns = ['pool','lines_reserved','day_of_week','month','minute_of_day','year','reserved_Vodnik'] 

	@abstractmethod
	def build_model(self):
		"""
		Abstrac function that builds model. Must be implemented by child class.
		In here should be implemented neural network keras model in self.model
		"""
		pass

	def fit_with_training_data(self, epochs=10, validation_split=0.3, verbose=1, use_cpu=True, test_mse=True):
		"""
		Trains model for defined number of epochs on training data.
		:param epochs: Number of epochs for fitting.
		:param validation_split: Percentage of training data that will be used for validation
		:param verbose: Verbose settign of fit function
		:param use_cpu: If True it will force to use CPU even if GPU is available
		:param test_mse: If True mean square error on testing set will be tested after fitting is done and result save in history
		"""
		x_train, y_train, x_test, y_test = self.dh.generate_feature_vectors_for_cnn(self.columns, self.time_steps_back)
		self.fit(x_train, y_train, epochs, validation_split, verbose, use_cpu, test_mse)

	def fit(self, x_train, y_train, epochs=10, validation_split=0.3, verbose=1, use_cpu=True, test_mse=True):
		"""
		Trains model for defined number of epochs on provided data.
		:param x_train: Numpy array with training data
		:param y_train: Numpy array with ground truth results for training data
		:param epochs: Number of epochs for fitting.
		:param validation_split: Percentage of training data that will be used for validation
		:param verbose: Verbose settign of fit function
		:param use_cpu: If True it will force to use CPU even if GPU is available
		:param test_mse: If True mean square error on testing set will be tested after fitting is done and result save in history
		"""
		if self.model is not None:
			if use_cpu:
				with tf.device('/cpu:0'):
					history = self.model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, verbose=verbose)
			else:
				history = self.model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, verbose=verbose)
			self.update_fit_history(history, test_mse)
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

	def save_model(self, name_addition=None):
		"""
		Saves model and weights with name defined in model_name
		"""
		name = self.model_name
		if name_addition is not None:
			name += name_addition

		model_json = self.model.to_json()
		with open(name+'.json', 'w') as json_file:
		    json_file.write(model_json)

		self.model.save_weights(name+'_weights.h5')
		print('Model saved to disk with name: ' + name)

	def load_model(self, name_addition=None):
		"""
		Loads model and weights with name defined in model_name
		:param name_addition: 
		"""
		name = self.model_name
		if name_addition is not None:
			name += name_addition

		json_file = open(name+'.json', 'r')
		model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(model_json)
		self.model.load_weights(name+'_weights.h5')
		print('Model %s loaded from disk'%(name))

	def setup(self, columns, time_steps_back):
		"""
		Trains model for defined number of epochs on provided data.
		:param columns: Colmuns to keep in data
		:param time_steps_back: Number of time steps used for prediction
		"""
		self.columns = columns
		self.time_steps_back = time_steps_back
		self.build_model()

	def get_mse(self):
		"""
		Computes mean square error on model
		:return: mean square error
		"""
		return self.dh.mse_on_testing_days(self.model, self.columns, self.time_steps_back, True)

	def show_n_predictions(self, n):
		"""
		Plots `n` predictions from training data using this model.
		:param n: If integer then represents number of random testing days. If list of integers
					then represents day ids from testing days. Last possible option is 
					string `all` that will plot all testing days.
		"""
		self.dh.show_n_days_prediction(self.model, self.columns, n, self.time_steps_back, True)

	def print_mse(self):
		"""
		Prints mean square error
		"""
		print('\nMSE = %.2f\n'%(self.get_mse()))

	def update_fit_history(self, history, test_mse):
		"""
		Adds progress of validation loss and training loss to `fit_history`
		If test_mse is True than also tests MSE and adds result to `fit_history`
		:param history: Keras fit history object
		:param test_mse: Flag if MSE should be tested
		"""
		if 'val_loss' in history.history.keys():
			for value in history.history['val_loss']:
				self.fit_history['val_loss'].append(value)
		if 'loss' in history.history.keys():
			for value in history.history['loss']:
				self.fit_history['loss'].append(value)
		if test_mse:
			mse = self.get_mse()
			if len(self.fit_history['mse']) > 0 and mse < min(self.fit_history['mse']):
				name_addition = '_MSE_%.0f'%(mse)
				self.save_model(name_addition)
				print('\nNew best MSE = %.2f\n'%(mse))
			else:
				print('\nMSE = %.2f\n'%(mse))
			self.fit_history['mse'].append(mse)			
