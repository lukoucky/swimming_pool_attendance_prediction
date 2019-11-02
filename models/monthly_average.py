from days_statistics import DaysStatistics
import numpy as np
import pickle
import os

class MonthlyAverageClassifier():
	"""
	Monthly average classifier. Using DaysStatistics to compute average
	attance for each month. Each monthly average is also split to two
	averages - one for weekend and one for weekday.
	"""
	def __init__(self):
		"""
		MonthlyAverageClassifier constructor.
		"""
		self.ds = DaysStatistics()
		self.columns = None
		self.time_steps_back = 3

	def fit(self, days_list, columns=None, time_steps_back=3):
		"""
		Fits MonthlyAverageClassifier. Fitting is done by generating monthly averages
		from training dataset in DaysStatistics class.
		:param days_list: List of days that should be used for average computation
		:param columns: List of columns that should remain in generated features. Deafult is None, when 
                        all columns appart from `time` remains.
        :param time_step_back: Number of time stamps in history that are packed together as input features
        """
		self.ds.days = days_list
		self.ds.generate_averages('data/monthly_avg_classifier_days_statistics.pickle', False)
		self.columns = columns
		self.time_steps_back = time_steps_back

		if columns is not None:
			for i, name in enumerate(self.columns):
				if name == 'month':
					self.month_id = - len(self.columns) + i
				if name == 'hour':
					self.hour_id = - len(self.columns) + i
				if name == 'minute':
					self.minute_id = - len(self.columns) + i
				if name == 'day_of_week':
					self.day_of_week_id = - len(self.columns) + i
		else:
			self.month_id = -53
			self.hour_id = -51
			self.minute_id = -50
			self.day_of_week_id = -54

	def predict(self, x):
		"""
		Predicts future attandance one step into to the future.
		So far works only for single feature input.
		:param x: Input feature vector
		:return: One step into future prediction
		"""
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

