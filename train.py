import os 
import pandas as pd
import numpy as np
import pickle
import datetime
import numpy as np
from random import Random
from utils import Day
from configuration import *
from models.weighted_average import WeightedAverage
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def trim_dataset(data, batch_size):
	"""
	Trims dataset to a size that's divisible by BATCH_SIZE
	:param data: D
	:param batch_size:
	:return:
	"""
	no_of_rows_drop = data.shape[0]%batch_size
	if(no_of_rows_drop > 0):
		return data[:-no_of_rows_drop]
	else:
		return data


def generate_days_list(data_frame, pickle_export_path=None):
	"""

	"""
	data = list()
	last_day = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d')
	last_ts = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d')

	for index, row in data_frame.iterrows():
		ts = datetime.datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
		if ts.date() > last_day.date():
			last_day = datetime.datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
			last_day = last_day.replace(hour=0, minute=0, second=0)
			day = Day(ts)
			if len(data) > 0:
				data[-1].end_index = index - 1
				data[-1].data = data_frame.iloc[data[-1].begin_index:index]

				if data[-1].close_index == 0:
					data[-1].close_index = index - 1

				if data[-1].open_index == 0:
					data[-1].open_index = data[-1].begin_index

			day.begin_index = index
			data.append(day)
		else:
			day_of_week = int(row['day_of_week'])
			if day_of_week < 5:
				if ts.hour > 5 and last_ts.hour < 6:
					day.open_index = index
			else:
				if ts.hour > 9 and last_ts.hour < 10:
					day.open_index = index

			if ts.hour > 21 and last_ts.hour < 22:
				day.close_index = index
		
		last_ts = ts

	data = data[:-2]
	data[-1].end_index = index
	if data[-1].close_index == 0:
		data[-1].close_index = index
		data[-1].data = data_frame.iloc[data[-1].begin_index:data[-1].end_index]

	clean_data = list()
	for d in data:
		if d.end_index - d.begin_index > 10:
			clean_data.append(d)	


	data_ids = list(range(0,len(clean_data)))
	Random(RANDOM_SEED).shuffle(data_ids)

	n_rows = len(data_ids)
	n_train_data = int(n_rows * 0.4)
	n_validation_data = int(n_rows * 0.2)

	train_data = [clean_data[i] for i in data_ids[0:n_train_data]]
	validation_data = [clean_data[i] for i in data_ids[n_train_data:n_train_data+n_validation_data]]
	test_data = [clean_data[i] for i in data_ids[n_train_data + n_validation_data:]]


	if pickle_export_path is not None:
		with open(pickle_export_path, 'wb') as f:
			data = [train_data, validation_data, test_data]
			pickle.dump(data, f)

	return train_data, validation_data, test_data


def predict_day(predictor, day):
	x, y = day.build_timeseries()
	y_pred = list()
	for i, data in enumerate(x):
		if i > 3:
			data[0] = y_pred[-3]
			data[112] = y_pred[-2]
			data[224] = y_pred[-1]
		prediction = predictor.predict([data])
		y_pred.append(prediction[0])
	return y_pred


def predict_day_without_reservations(predictor, day):
	x, y = day.build_timeseries_without_reservations()
	y_pred = list()
	for i, data in enumerate(x):
		if i > 3:
			data[0] = y_pred[-3]
			data[13] = y_pred[-2]
			data[26] = y_pred[-1]
		prediction = predictor.predict([data])
		y_pred.append(prediction[0])
	return y_pred


def build_feature_vector(days_list, without_reserved=False):
	x = list()
	y = list()
	for day in days_list:
		if without_reserved:
			x_day, y_day = day.build_timeseries_without_reservations()
		else:
			x_day, y_day = day.build_timeseries()
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


def get_data(pickle_path=None):
	"""
	Loads training, validation and testing data. If `pickle_path` is
	path to pickle file data are loaded from this file. Otherwise are data
	loaded from csv file in DATASET_CSV_PATH
	:param pickle_path: Path to pickle with processed data
	:return: thre lists contating Day instances. First for traning, second for validation and third for testing
	"""
	if pickle_path is None or not os.path.isfile(pickle_path):
		train_data = pd.read_csv(DATASET_CSV_PATH)
		d_train, d_val, d_test = generate_days_list(train_data, pickle_path)
	else:
		with open(pickle_path, 'rb') as input_file:
			d_train, d_val, d_test = pickle.load(input_file)

	d_train = clean_data(d_train)
	d_val = clean_data(d_val)
	d_test = clean_data(d_test)

	return d_train, d_val, d_test


def clean_data(day_list, zero_attandance_limit=5):
	clean_list = list()
	for day in day_list:
		if day.get_zero_attandance_during_open_hours() <= zero_attandance_limit:
			clean_list.append(day)
	return clean_list

def train_random_forrest(x, y, n_estimators):
	x_len = x.shape[1]
	pickle_name = 'data/random_forrest_n%d_%d' % (n_estimators, x_len)
	if os.path.isfile(pickle_name):
		with open(pickle_name, 'rb') as f:
			clf = pickle.load(f)
		return clf
	else:
		clf = RandomForestClassifier(n_estimators=n_estimators)
		clf.fit(x, y)
		with open(pickle_name, 'wb') as f:
			pickle.dump(clf, f)
		return clf

def train_adaboost(x, y, n_estimators):
	x_len = x.shape[1]
	pickle_name = 'data/adaboost_n%d_%d' % (n_estimators, x_len)
	if os.path.isfile(pickle_name):
		with open(pickle_name, 'rb') as f:
			clf = pickle.load(f)
		return clf
	else:
		clf = AdaBoostClassifier(n_estimators=n_estimators)
		clf.fit(x, y)
		with open(pickle_name, 'wb') as f:
			pickle.dump(clf, f)
		return clf

if __name__ == '__main__':
	d_train, d_val, d_test = get_data('data/days.pickle')

	# x_train_s, y_train_s = build_feature_vector(d_train, True)
	# x_test_s, y_test_s = build_feature_vector(d_test, True)

	# x_train, y_train = build_feature_vector(d_train)
	# x_test, y_test = build_feature_vector(d_test)

	# data = [x_train, y_train, x_test, y_test]
	# data_s = [x_train_s, y_train_s, x_test_s, y_test_s]

	# with open('data/data.pickle', 'wb') as f:
	# 	pickle.dump(data, f)

	# with open('data/data_s.pickle', 'wb') as f:
	# 	pickle.dump(data_s, f)

	with open('data/data.pickle', 'rb') as f:
		x_train, y_train, x_test, y_test = pickle.load(f)

	with open('data/data_s.pickle', 'rb') as f:
		x_train_s, y_train_s, x_test_s, y_test_s = pickle.load(f)

	clf = train_random_forrest(x_train, y_train, 10)
	clf_s = train_random_forrest(x_train_s, y_train_s, 10)

	# clf = train_adaboost(x_train, y_train, 50)
	# clf_s = train_adaboost(x_train_s, y_train_s, 50)
	# clf = SVC(gamma='auto')

	day_id = 56
	x, y = d_test[day_id].build_timeseries()
	y_pred = predict_day(clf, d_test[day_id])
	y_pred_s = predict_day_without_reservations(clf_s, d_test[day_id])

	l1, = plt.plot(y, label='GT')
	l2, = plt.plot(y_pred, label='Full dataset')
	l3, = plt.plot(y_pred_s, label='No reserve')
	plt.legend(handles=[l1, l2, l3])
	plt.show()	

