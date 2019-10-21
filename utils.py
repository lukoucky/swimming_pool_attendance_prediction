import datetime
import numpy as np

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

	def data_drame_to_timeseries_numpy(self, data, time_step=3):
		"""
		Creates vector of prediction results and features.
		:param data: DataFrame 
		:param time_step: Number of time stamps packed together as input features
		:return: Two numpy arrays with features and results
		"""
		matrix = data.to_numpy()
		dim_0 = matrix.shape[0] - time_step
		dim_1 = matrix.shape[1]

		x = np.zeros((dim_0, time_step*dim_1))
		y = np.zeros((dim_0,))
		
		for i in range(dim_0):
			x_data = matrix[i:time_step+i]
			x[i] = np.reshape(x_data,x_data.shape[0]*x_data.shape[1])
			y[i] = matrix[time_step+i, 0]
		return x, y

	def build_timeseries(self, time_step=3):
		"""
		Creates vector of prediction results and features.
		:param time_step: Number of time stamps packed together as input features
		:return: Two numpy arrays with features and results
		"""
		clean_data = self.data.copy()
		clean_data.drop(columns=['time'], inplace=True)
		return self.data_drame_to_timeseries_numpy(clean_data, time_step)

	def build_timeseries_without_reservations(self, time_step=3):
		"""
		Creates vector of prediction results and features without reservation organisations.
		:param time_step: Number of time stamps packed together as input features
		:return: Two numpy arrays with features and results
		"""
		reservation_columns = ['time']
		clean_data = self.data.copy()
		for column in self.data:
			if column.startswith('reserved_'):
				reservation_columns.append(column)
		clean_data.drop(columns=reservation_columns, inplace=True)
		return self.data_drame_to_timeseries_numpy(clean_data, time_step)

	def get_zero_attandance_during_open_hours(self):
		"""
		Counts how many data samples from open hours have zero attandance in pool.
		:return: Number of zeros in `pool` column between open_index and close_index
		"""
		start = self.open_index - self.begin_index
		stop = self.close_index - self.begin_index
		attandance = self.data['pool'][start:stop]
		return sum(attandance == 0)

	def __repr__(self):
		return 'ts: %s, %d - %d - %d - %d' % (self.ts.strftime('%Y-%m-%d %H:%M:%S'), self.begin_index, self.open_index, self.close_index, self.end_index)
