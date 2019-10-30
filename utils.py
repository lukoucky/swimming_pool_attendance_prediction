import datetime
import numpy as np
import inspect
import itertools
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from models.random_forest_classifier import RandomForest
import pickle
import matplotlib.pyplot as plt
import os

class WeatherData(object):
	"""
	WeatherData object holds weather measurements at particular time stamp from all available measurement stations.
	"""
	def __init__(self, ts, default_station):
		"""
		Constructor stores given time stamp `ts` and creates empty dictionary that will be filled with weather data
		by setters.
		:param ts: datetime time stamp
		:param default_station: Index number of station thath will be used
					for for getters
		"""
		self.ts = ts
		self.default_station = default_station
		self.data = dict()
		self.stations = list()

	def set_data_from_database(self, data):
		"""
		Sets data response from database. Data can contain multiple measurements 
		from the same station due to the way they are collected from database.
		Only closest measurements to time stamp from each station will remain and other
		measurements will be ignored.
		:param data: DataFrame with response from database
		"""
		for index, row in data.iterrows():
			station = row['station']
			if station not in self.stations:
				self.stations.append(station)
				self.data[station] = Measurement(row)

		self.default_station = min(self.stations)

	def get_temperature(self):
		"""
		Returns temperature for station selected in default_station
		:return: Temperature at time stamp ts
		"""
		return self.data[self.default_station].temperature

	def get_wind(self):
		"""
		Returns wind speed for station selected in default_station
		:return: Wind spped at time stamp ts
		"""
		return self.data[self.default_station].wind

	def get_humidity(self):
		"""
		Returns humidity for station selected in default_station
		:return: Humidity at time stamp ts
		"""
		return self.data[self.default_station].humidity

	def get_precipitation(self):
		"""
		Returns precipitation for station selected in default_station
		:return: Precipitation at time stamp ts
		"""
		return self.data[self.default_station].precipitation

	def get_pressure(self):
		"""
		Returns air pressure for station selected in default_station
		:return: Air pressure at time stamp ts
		"""
		return self.data[self.default_station].pressure


class Measurement(object):
	"""
	Measurement holds weather information about temperature, wind,
	humidity, precipitation, pressure
	"""
	def __init__(self, dataframe_row):
		self.temperature = dataframe_row['temperature']
		self.wind = dataframe_row['wind']
		self.humidity = dataframe_row['humidity']
		self.precipitation = dataframe_row['precipitation']
		self.pressure = dataframe_row['pressure']

	def __repr__(self):
		s = 'temperature = %d, wind = %d, humidity = %d, precipitation = %d, pressure = %d' % (self.temperature, self.wind, self.humidity, self.precipitation, self.pressure)
		return s


class Day(object):
	"""
	Helper class that holds information about one day in dataset.
	It contains indices where day starts and ends in dataset DataFrame.
	"""
	def __init__(self, ts):
		"""
		Constructor for Day class
		:param ts: Full time stamp 
		"""
		self.ts = ts
		self.begin_index = 0
		self.end_index = 0
		self.open_index = 0
		self.close_index = 0
		self.data = None
		self.timeseries_export_ignore = ['time', 'humidity', 'pressure']

	def data_frame_to_timeseries_numpy(self, data, time_step_back=3, time_stap_forward=1):
		"""
		Creates vector of prediction results and features.
		:param data: DataFrame 
		:param time_step_back: Number of time stamps packed together as input features
		:return: Two numpy arrays with features and results
		"""
		matrix = data.values
		dim_0 = matrix.shape[0] - time_step_back
		dim_1 = matrix.shape[1]

		x = np.zeros((dim_0, time_step_back*dim_1))
		y = np.zeros((dim_0, time_stap_forward))
		
		for i in range(dim_0):
			x_data = matrix[i:time_step_back+i]
			x[i] = np.reshape(x_data, x_data.shape[0]*x_data.shape[1])

			end_id = time_step_back+i+time_stap_forward
			if end_id >= dim_0:
				end_id = dim_0 - 1

			y_data = matrix[time_step_back+i:end_id, 0]
			if len(y_data) < time_stap_forward:
				y_provis = np.zeros(time_stap_forward,)
				for i, value in enumerate(y_data):
					y_provis[i] = value
				y[i] = y_provis
			else:
				y[i] = np.reshape(y_data, time_stap_forward)

		return x, y

	def build_timeseries(self, time_step=3, normalize=False):
		"""
		Creates vector of prediction results and features.
		:param time_step: Number of time stamps packed together as input features
		:param normalize: True if timeseries should be normalized
		:return: Two numpy arrays with features and results
		"""
		if normalize:
			clean_data = self.get_normalized_data()
		else:
			clean_data = self.data.copy()
		clean_data.drop(columns=['time'], inplace=True)
		return self.data_frame_to_timeseries_numpy(clean_data, time_step)

	def build_timeseries_without_reservations(self, time_step=3, normalize=False):
		"""
		Creates vector of prediction results and features without reservation organisations.
		:param time_step: Number of time stamps packed together as input features
		:param normalize: True if timeseries should be normalized
		:return: Two numpy arrays with features and results
		"""
		reservation_columns = ['time']
		if normalize:
			clean_data = self.get_normalized_data()
		else:
			clean_data = self.data.copy()
		for column in self.data:
			if column.startswith('reserved_'):
				reservation_columns.append(column)
		clean_data.drop(columns=reservation_columns, inplace=True)
		return self.data_frame_to_timeseries_numpy(clean_data, time_step)

	def get_zero_attandance_during_open_hours(self):
		"""
		Counts how many data samples from open hours have zero attandance in pool.
		:return: Number of zeros in `pool` column between open_index and close_index
		"""
		start = self.open_index - self.begin_index
		stop = self.close_index - self.begin_index
		attandance = self.data['pool'][start:stop]
		return sum(attandance == 0)

	def get_normalized_data(self):
		"""
		Normalizes all numeric columns in self.data.
		:return: Normalized self.data
		"""
		df = self.data.copy()

		df[df['pool'] < 0] = 0
		df[df['pool'] > 400] = 400
		df['pool'] = df['pool']/400
		
		df['lines_reserved'] = df['lines_reserved']/8
		df['day_of_week'] = df['day_of_week']/6
		df['month'] = df['month']/12
		df['day'] = df['day']/31
		df['hour'] = df['hour']/24
		df['minute'] = df['minute']/60

		for column in self.data.columns:
			if column.startswith('reserved_'):
				df[column] = df[column]/8

		if 'temperature' in self.data.columns:
			df[df['temperature'] < -45] = -45
			df[df['temperature'] > 45] = 45
			df['temperature'] = (df['temperature']+45.0)/90.0

		if 'wind' in self.data.columns:
			df[df['wind'] < 0] = 0
			df[df['wind'] > 100] = 100
			df['wind'] = df['wind']/100

		if 'humidity' in self.data.columns:
			df[df['humidity'] < 0] = 0
			df[df['humidity'] > 100] = 100
			df['humidity'] = df['humidity']/100

		if 'precipitation' in self.data.columns:
			df[df['precipitation'] < 0] = 0
			df[df['precipitation'] > 100] = 100
			df['precipitation'] = df['precipitation']/100

		if 'pressure' in self.data.columns:
			df[df['pressure'] < 800] = 800
			df[df['pressure'] > 1200] = 1200
			df['pressure'] = (df['pressure']-800)/400

		if 'temperature_binned' in self.data.columns:
			df['temperature_binned'] = df['temperature_binned']/7

		if 'wind_binned' in self.data.columns:
			df['wind_binned'] = df['wind_binned']/5

		if 'humidity_binned' in self.data.columns:
			df['humidity_binned'] = df['humidity_binned']/4

		if 'precipitation_binned' in self.data.columns:
			df['precipitation_binned'] = df['precipitation_binned']/3

		if 'pressure_binned' in self.data.columns:
			df['pressure_binned'] = df['pressure_binned']/4

		return df


	def __repr__(self):
		return 'ts: %s, %d - %d - %d - %d' % (self.ts.strftime('%Y-%m-%d %H:%M:%S'), self.begin_index, self.open_index, self.close_index, self.end_index)

class MyGridSearch:
	def __init__(self, estimator_name, param_dict):
		inspector = inspect.getargspec(eval(estimator_name))
		for key in param_dict.keys():
			assert key in inspector.args, 'Argument %s is not valid for class %s' % (key, estimator.__class__.__name__)

		self.estimator_name = estimator_name
		self.param_dict = param_dict
		self.evaluation = []
		self.parameters = []
		self.best_parameters = None
		self.best_evaluation = None
		self.best_estimator = None
		self.arguments = None
		self.generator = None
		self.grid_size = 1
		self.prepare_generator()
		self.utils_model = RandomForest()

	def fit(self, without_reserved=False):
		n = 1
		self.utils_model.without_reserved = without_reserved
		x_train, y_train, _, _= self.utils_model.load_data(without_reserved)
		for values in self.generator:
			params = {}
			estimator_str = self.estimator_name + '('
			for i in range(len(self.arguments)):
				estimator_str += self.arguments[i] + '=' + str(values[i]) + ', '
				params[self.arguments[i]] = values[i]
			estimator_str = estimator_str[:-2] + ')'

			self.parameters.append(params)
			e = eval(estimator_str)
			e.fit(x_train, y_train)
			score = self.utils_model.get_testing_mse(e)

			if len(self.evaluation) == 0 or score < min(self.evaluation):
				self.best_evaluation = score
				self.best_estimator = e
				self.best_parameters = params

			self.evaluation.append(score)
			print('%d out of %d done for parameters %s with score %.3f, best MSE so far = %.3f' % (n, self.grid_size, estimator_str, score, self.best_evaluation))
			n +=1 
		print('GridSearch for %s done.\nBest MSE = %.3f for parameters:' % (self.estimator_name, self.best_evaluation))
		print(self.best_parameters)
		print('Saving best estimator')
		est_path = 'data/%s_MyGridSearch_MSE_%.3f.pickle' % (self.estimator_name, self.best_evaluation)
		with open('data/RandomForestClassifier_RandomSearch.pickle', 'wb') as f:
			pickle.dump(self.best_estimator, f)

	def prepare_generator(self):
		self.arguments = list()
		product_str = 'itertools.product(['
		for key, value in self.param_dict.items():
			self.grid_size *= len(value)
			self.arguments.append(key)
			for element in value:
				product_str += str(element) + ', '
			product_str = product_str[:-2] + '], ['
		product_str = product_str[:-3] + ')'
		self.generator = eval(product_str)


class DaysStatistics:
	def __init__(self, days_list):
		self.days = days_list
		self.averages_weekday = list()
		self.averages_weekend = list()
		self.generate_averages()

	def generate_averages(self, override_pickle=False):
		pickle_path = 'data/days_statistics.pickle'
		if os.path.isfile(pickle_path) and not override_pickle:
			with open(pickle_path, 'rb') as input_file:
				self.averages_weekday, self.averages_weekend = pickle.load(input_file)
		else:
			for month in range(12):
				self.averages_weekday.append([])
				for i in range(288):
					self.averages_weekday[month].append(0)

			self.averages_weekend = copy.deepcopy(self.averages_weekday)
			n_weekday = copy.deepcopy(self.averages_weekday)
			sums_weekday = copy.deepcopy(self.averages_weekday)
			n_weekend = copy.deepcopy(self.averages_weekday)
			sums_weekend = copy.deepcopy(self.averages_weekday)
			for day in self.days:
				for index, row in day.data.iterrows():
					month = row['month']-1
					day_id = self.get_list_id(row['hour'], row['minute'])
					if row['day_of_week'] < 5:
						sums_weekday[month][day_id] += row['pool']
						n_weekday[month][day_id] += 1
					else:
						sums_weekend[month][day_id] += row['pool']
						n_weekend[month][day_id] += 1

			for month in range(12):
				for i in range(288):
					if n_weekday[month][i] > 0:
						self.averages_weekday[month][i] = sums_weekday[month][i]/n_weekday[month][i]
					if n_weekend[month][i] > 0:
						self.averages_weekend[month][i] = sums_weekend[month][i]/n_weekend[month][i]

			with open(pickle_path, 'wb') as f:
				pickle.dump([self.averages_weekday, self.averages_weekend], f)

	def get_average_for_month(self, month, weekend):
		if weekend:
			return self.averages_weekend[month]
		else:
			return self.averages_weekday[month]

	def get_average_for_month_at_time(self, month, hour, minute, weekend):
		if weekend:
			return self.averages_weekend[month][self.get_list_id(hour, minute)]
		else:
			return self.averages_weekday[month][self.get_list_id(hour, minute)]

	def get_month_average_for_time(self, month, hour, minute):
		day_id = self.get_list_id(hour, minute)
		return self.averages_weekday[month][day_id]

	def get_average_for_last_days(self, n_days, month, day, hour, minute):
		pass 

	def get_average_for_last_days_at_time(self, n_days, month, day, hour, minute):
		pass 

	def get_list_id(self, hour, minute):
		return hour*12 + minute//5

	def plot_year_averages_by_month(self, weekend):
		plot_name = 'Average monthly attandance for '
		if weekend:
			plot_name += 'weekends'
		else:
			plot_name += 'weekdays'

		fig = plt.figure(plot_name)
		for i in range(12):
			if weekend:
				data = self.averages_weekend[i]
			else:
				data = self.averages_weekday[i]

			ax = fig.add_subplot(4, 3, i+1)
			ax.title.set_text('Month %d' % (i+1))
			l1, = plt.plot(data)
			ax.grid(True)
			ax.set_ylim([0,300])
		plt.show()

	def plot_monthly_average(self, month, weekend, other_data=None, other_data_offset=48):
		plot_name = 'Average attandance for '
		if weekend:
			plot_name += 'weekends'
			data = self.averages_weekend[month]
		else:
			plot_name += 'weekdays'
			data = self.averages_weekday[month]

		fig = plt.figure(plot_name + 'month ' + str(month))	
		ax = fig.add_subplot(1,1,1)
		l1, = plt.plot(data)
		if other_data is not None:
			y = range(other_data_offset,other_data_offset+len(other_data))
			l2, = plt.plot(y, other_data)

		ax.grid(True)
		ax.set_ylim([0,300])
		plt.show()

