import copy
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime

class DaysStatistics:
	def __init__(self):
		self.days = list()
		self.averages_weekday = list()
		self.averages_weekend = list()
		self.bad_dates = ['2018-02-20','2018-06-05','2018-06-06','2018-06-07','2018-06-08','2018-06-11','2018-06-12','2018-06-13','2018-06-14','2018-09-05','2018-03-17','2018-05-05','2018-06-10','2018-12-01']
		self.generate_averages()

	def generate_averages(self, pickle_path='data/days_statistics.pickle', override_pickle=False):
		if os.path.isfile(pickle_path) and not override_pickle:
			with open(pickle_path, 'rb') as input_file:
				self.averages_weekday, self.averages_weekend = pickle.load(input_file)
		else:
			n_weekday = list()
			sums_weekday = list()
			n_weekend = list()
			sums_weekend = list()
			for month in range(12):
				self.averages_weekday.append([])
				self.averages_weekend.append([])
				n_weekday.append([])
				sums_weekday.append([])
				n_weekend.append([])
				sums_weekend.append([])
				for i in range(288):
					self.averages_weekday[month].append(0)
					self.averages_weekend[month].append(0)
					n_weekday[month].append(0)
					sums_weekday[month].append(0)
					n_weekend[month].append(0)
					sums_weekend[month].append(0)

			for day in self.days:
				ts = datetime.strptime(day.data['time'].iloc[0], '%Y-%m-%d %H:%M:%S')
				if ts.strftime('%Y-%m-%d') not in self.bad_dates:
					for index, row in day.data.iterrows():
						month = row['month']-1
						day_id = self.get_list_id(row['hour'], row['minute'])
						if row['day_of_week'] < 5:
							sums_weekday[month][day_id] += int(row['pool'])
							n_weekday[month][day_id] += 1
						else:
							sums_weekend[month][day_id] += int(row['pool'])
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

		fig = plt.figure(plot_name, figsize=(19,12))
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
			ax.set_xlim([0,300])
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

	def generate_organisation_addition(self):
		reserved_columns = list()
		total_attandance = list()
		n = list()
		for column in self.days[0].data.columns:
			if column.startswith('reserved_'):
				reserved_columns.append(column)
				total_attandance.append(0)
				n.append(0)

		for day in self.days:
			for index, row in day.data.iterrows():
				for i, column in enumerate(reserved_columns):
					if int(row[column]) > 0:
						weekend = True
						if int(row['day_of_week']) < 5:
							weekend = False
						total_attandance[i] += row['pool'] - self.get_average_for_month_at_time(int(row['month'])-1, int(row['hour']), int(row['minute']), weekend)
						n[i] += 1

		self.org_addition = dict()
		for i, column in enumerate(reserved_columns):
			if n[i] > 0:
				self.org_addition[column] = total_attandance[i]/n[i]
			else:
				self.org_addition[column] = 0


if __name__ == '__main__':
	with open('data/days.pickle', 'rb') as input_file:
		d1,d2,d3 = pickle.load(input_file)

	ds = DaysStatistics()
	ds.days = d1+d2
	ds.averages_weekday = list()
	ds.averages_weekend = list()
	ds.generate_averages(override_pickle=True)
	# print(ds.org_addition)
	ds.plot_monthly_average(11, True)

