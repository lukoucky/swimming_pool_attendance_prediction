from days_statistics import DaysStatistics
import numpy as np
import pickle
import os

class MonthlyAverageClassifier():
	def __init__(self):
		self.ds = DaysStatistics()
		self.columns = None
		self.time_steps_back = 3

	def fit(self, days_list, columns=None, time_steps_back=3):
		self.ds.days = days_list
		self.ds.generate_averages('data/monthly_avg_classifier_days_statistics.pickle', True)
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

		self.ds.plot_year_averages_by_month(True)

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

	# def predict(self, x):
	# 	x = x[0]
	# 	print(x)
	# 	print(len(x.shape))
	# 	y_pred = np.zeros(x.shape[0], 1)
	# 	for i, row in enumerate(x):
	# 		hour = int(row[self.hour_id])
	# 		minute = int(row[self.minute_id]) + 5
	# 		if minute > 59:
	# 			hour += 1
	# 			minute -= 60

	# 		print('ids', self.month_id, self.hour_id, self.minute_id)
	# 		print('Predicting for ',int(row[self.month_id]), hour, minute)
	# 		y_pred[i] = self.ds.get_month_average_for_time(int(row[self.month_id])-1, hour, minute)
	# 		print(y_pred)
	# 		print('\n\n')
	# 	return y_pred.ravel()
