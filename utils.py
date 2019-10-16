import datetime


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



