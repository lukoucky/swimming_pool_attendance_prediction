from models.model import Model
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import os

class MonthlyAverage():
	def __init__(self):
		super().__init__(MyAverageModel())

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

class MyAverageModel():
	"""
	Average model that implements functions called by base class Model
	"""
	def __init__(self):
		self.averages_weekday = list()
		self.averages_weekend = list()
		self.fitted = False

	def fit(self, days_list):
		for month in range(12):
			self.averages_weekday.append([])
			for i in range(288):
				self.averages_weekday[month].append(0)

		self.averages_weekend = copy.deepcopy(self.averages_weekday)
		n_weekday = copy.deepcopy(self.averages_weekday)
		sums_weekday = copy.deepcopy(self.averages_weekday)
		n_weekend = copy.deepcopy(self.averages_weekday)
		sums_weekend = copy.deepcopy(self.averages_weekday)
		for day in days_list:
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

		self.fitted = True

	def predict(self, x):
		pass
