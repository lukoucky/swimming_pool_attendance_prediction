from abc import ABC, abstractmethod
import os
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import random
from utils import DaysStatistics

class Model(ABC):
	def __init__(self, classificator):
		self.model = classificator
		self.random_param_search = None
		self.are_parameters_tuned = False
		self.without_reserved = False
		d_train, d_val, d_test = self.get_all_days_lists()
		self.days_stats = DaysStatistics(d_train + d_val)

	@abstractmethod
	def fit(self, without_reserved=False):
		pass

	@abstractmethod
	def predict(self, data):
		pass

	@abstractmethod
	def tune_parameters(self, data_x, data_y):
		pass

	@abstractmethod
	def load_tuned_parameters(self, pickle_path=None):
		pass

	def load_data(self, without_reserved=False, normalize=False):
		data_path = 'data/data'
		if self.without_reserved or without_reserved:
			data_path += '_short'
		if normalize:
			data_path += '_normalize'
		data_path += '.pickle'

		if os.path.isfile(data_path):
			with open(data_path, 'rb') as f:
				x_train, y_train, x_test, y_test = pickle.load(f)
		else:
			d_train, d_val, d_test = self.get_all_days_lists()
			x_train, y_train = self.build_feature_vector(d_train+d_val, without_reserved, normalize)
			x_test, y_test = self.build_feature_vector(d_test, without_reserved, normalize)
			data = [x_train, y_train, x_test, y_test]
			with open(data_path, 'wb') as f:
					pickle.dump(data, f)

		return x_train, y_train, x_test, y_test

	def build_feature_vector(self, days_list, without_reserved=False, normalize=False):
		x = list()
		y = list()
		for day in days_list:
			if self.without_reserved or without_reserved:
				x_day, y_day = day.build_timeseries_without_reservations(3, normalize)
			else:
				x_day, y_day = day.build_timeseries(3, normalize)
			x.append(x_day)
			y.append(y_day)

		for i, data in enumerate(x):
			if i > 0:
				x_data = np.concatenate([x_data,data])
			else:
				x_data = np.array(data)

		for i, data in enumerate(y):
			if i > 0:
				y_data = np.concatenate([y_data,data])
			else:
				y_data = np.array(data)

		return x_data, y_data

	def build_feature_vector_with_average(self, days_list, without_reserved=False, normalize=False):
		x, y = self.build_feature_vector(days_list, without_reserved, normalize)
		montly_averages = np.zeros((x.shape[0],1))
		for i, row in enumerate(x):
			montly_averages[i] = self.get_monthly_average_for_feature(row)

		new_x = np.concatenate((x, montly_averages),  axis=1)
		return new_x, y

	def get_monthly_average_for_feature(self, row):
		hour = int(row[-51])
		minute = int(row[-50]) + 5

		weekend = False
		if int(row[-54]) > 4:
			weekend = True

		if minute > 60:
			minute -= 60
			hour += 1
		return self.days_stats.get_average_for_month_at_time(int(row[-53])-1, hour, minute, weekend)

	def shuffle_date(self, x, y):
		if len(x) == len(y):
			p = np.random.RandomState(seed=17).permutation(len(x))
			return x[p], y[p]
		else:
			print('ERROR (shuffle_data) - length of x and y are different (%d vs %d)' % (len(x), len(y)))
			return x, y

	def get_data_without_reservations(self, normalize=False):
		return self.load_data(True, normalize)

	def get_data_without_columns(self, filter_columns):
		data_columns = self.get_all_columns()
		for column in filter_columns:
			if column in data_columns:
				data_columns.remove(column)
			else:
				print('Warning: column `%s` is not in data' %  (column))
		# TODO get all data and remove columns

	def get_all_columns(self):
		d_train, d_val, d_test = self.get_all_days_lists()
		return d_train[0].data.columns

	def show_n_days_prediction(self, days=6, model=None):
		d_train, d_val, d_test = self.get_all_days_lists()
		x_train, y_train, x_test, y_test = self.load_data()
		if isinstance(days, list):
			days_list = days
		elif isinstance(days, int):
			n_data = len(d_test)
			days_list = random.sample(range(0, n_data), days)

		if model is None:
			model = self.model

		rows = days//2
		columns = 2
		if len(days_list) == 1:
			rows = 1
			columns = 1

		fig = plt.figure(self.__class__.__name__)
		for i, day_id in enumerate(days_list):
			x, y = d_test[day_id].build_timeseries()
			y_pred = self.predict_day(d_test[day_id])
			ax = fig.add_subplot(rows, columns, i+1)
			ax.title.set_text('mse=%.2f' % (mean_squared_error(y_pred, y)))
			l1, = plt.plot(y)
			l2, = plt.plot(y_pred)	
		plt.show()

	def get_testing_mse(self, model=None):
		d_train, d_val, d_test = self.get_all_days_lists()
		return self.mse_by_day(d_test, model)

	def mse_by_day(self, days_list, model=None):
		# preicts day by day and computes rmse for each day - not full x!
		# Generate histogram of MSEs in whole dataset per day
		errors = list()
		for day in days_list:
			x, y = day.build_timeseries()
			y_pred = self.predict_day(day, model)
			errors.append(mean_squared_error(y, y_pred))

		return sum(errors)

	def get_all_days_lists(self):
		DAYS_PICKLE_PATH = 'data/days.pickle'
		DATASET_CSV_PATH = 'data/dataset.csv'
		if not os.path.isfile(DAYS_PICKLE_PATH):
			train_data = pd.read_csv(DATASET_CSV_PATH)
			d_train, d_val, d_test = generate_days_list(train_data, DAYS_PICKLE_PATH)
		else:
			with open(DAYS_PICKLE_PATH, 'rb') as input_file:
				d_train, d_val, d_test = pickle.load(input_file)

		d_train = self.clean_data(d_train)
		d_val = self.clean_data(d_val)
		d_test = self.clean_data(d_test)

		return d_train, d_val, d_test

	def clean_data(self, day_list, zero_attandance_limit=5):
		clean_list = list()
		for day in day_list:
			if day.get_zero_attandance_during_open_hours() <= zero_attandance_limit:
				clean_list.append(day)
		return clean_list

	def tune_parameters_on_full_training_data(self):
		x_train, y_train, x_test, y_test = self.load_data()
		x, y = self.shuffle_date(x_train, y_train)
		self.tune_parameters(x, y)

	def set_best_tuned_parameters(self):
		if self.are_parameters_tuned:
			self.model = self.random_param_search.best_estimator_
		else:
			print('Warning: Cannot set best tuned parameters, tuning not done yet.')

	def evaluate_model_on_testing_data(self):
		x_train, y_train, x_test, y_test = self.load_data()
		self.evaluate(x_test, y_test)

	def evaluate(self, data_x, data_y):
		pred_y = self.model.predict(data_x)
		print('Evaluation MSE:', mean_squared_error(pred_y, data_y))

	def predict_day(self, day, predictor=None, without_reserved=False):
		if predictor == None:
			predictor = self.model

		if self.without_reserved or without_reserved:
			x, y = day.build_timeseries_without_reservations()
		else:
			x, y = day.build_timeseries()

		y_pred = list()
		for i, data in enumerate(x):
			if i > 3:
				data[0] = y_pred[-3]
				data[int(x.shape[1]/3)] = y_pred[-2]
				data[int((x.shape[1]/3)*2)] = y_pred[-1]
			########
			# TODO remove following line - works only for monthly average in feature
			#########
			pred_data = np.append(data,[self.get_monthly_average_for_feature(data)])
			prediction = predictor.predict([pred_data])
			y_pred.append(prediction[0])
		return y_pred

	def print_data(self, data):
		print(data[0],data[56],data[112],)
