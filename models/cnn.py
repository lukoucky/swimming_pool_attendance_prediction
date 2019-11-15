from data_helper import DataHelper
import numpy as np
import pickle
import os

class ConvolutionalNeuralNetwork():
	def __init__(self):
		self.dh = DataHelper()
		self.columns = ['pool','day_of_week','month','hour','minute'] 
		self.time_steps_back = 10

	def build_model(self):
		model = Sequential()
		model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Flatten())
		model.add(Dense(50, activation='relu'))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse')

	def fit(self):
		x_train, y_train, x_test, y_test = dh.generate_feature_vectors(columns, time_step_back)

	def predict(self, x):
		row = x[0]
		hour = int(row[self.hour_id])
		minute = int(row[self.minute_id]) + 5
		if minute > 59:
			hour += 1
			minute -= 60

		weekend = False
		if int(row[self.day_of_week_id]) > 4:
			weekend = True

		y_pred = self.ds.get_average_for_month_at_time(int(row[self.month_id])-1, hour, minute, weekend)
		return [y_pred]