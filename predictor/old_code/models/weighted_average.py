from models.model import Model
import datetime


class WeightedAverage(Model):

	def __init__(self, name):
		self.name = name
		self.weekday_window = 10
		self.weekend_window = 6
		self.train_data = None

	def fit(self, data):
		self.train_data = data

	def predict(self, time_stamp):

		date_str = time_stamp.strftime('%Y-%m-%d')

		print('Predict average')
