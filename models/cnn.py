from data_helper import DataHelper
import numpy as np
import pickle
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import model_from_json

class ConvolutionalNeuralNetwork():
	def __init__(self):
		self.dh = DataHelper()
		self.columns = ['pool','day_of_week','month','minute_of_day','year'] 
		self.time_steps_back = 3
		self.model = None
		self.build_model()

	def build_model(self):
		n_features = len(self.columns)
		self.model = Sequential()
		self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.time_steps_back, n_features)))
		self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Flatten())
		self.model.add(Dense(100, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer='adam', loss='mse')

	def fit(self, x_train, y_train):
		if self.model is not None:
			# x_train, y_train, x_test, y_test = self.dh.generate_feature_vectors_for_cnn(self.columns, self.time_steps_back)
			self.model.fit(x_train, y_train, epochs=10, validation_split=0.2)
			self.save_model()
		else:
			print('Cannot fit. Build model first.')

	def predict(self, x):
		return self.model.predict(x)

	def save_model(self):
		model_json = self.model.to_json()
		with open("cnn_model.json", "w") as json_file:
		    json_file.write(model_json)

		self.model.save_weights("cnn_weights.h5")
		print("CNN model saved to disk")

	def load_model(self):
		json_file = open('cnn_model.json', 'r')
		model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(model_json)
		self.model.load_weights("cnn_weights.h5")
