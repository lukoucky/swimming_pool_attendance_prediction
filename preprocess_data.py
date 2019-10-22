import sqlite3
import datetime
import pandas as pd
from utils import WeatherData
from random import Random
from configuration import *
import matplotlib.pyplot as plt


def get_data_from_database(query, db_path=DB_PATH):
	"""
	Executes query on SQLite database located at db_path and returns response as DataFrame
	:param query: query to be executed in string format
	:param db_path: path to SQLite db file, default DB_PATH constant
	:return: DataFrame with response
	"""
	connection = sqlite3.connect(db_path)
	return pd.read_sql_query(query, connection)


def get_all_occupancy_data(remove_closed=True):
	"""
	Reads all data from `occupancy` table in database and returns it as pandas DataFrame. Optional input argument
	`remove_closed` can be used to remove from data all time stamps where the pool is closed. This option
	is turned on by default. Setting remove_close to False will return DataFrame with all 24 hour per day data.
	:param remove_closed: Flag if data from hours when pool is closed should be removed
	:return: DataFrame with all occupancy data from database
	"""
	if remove_closed:
		query =	"""
				SELECT * FROM occupancy WHERE 
				(day_of_week < 5  AND SUBSTR(time, 12, 2) > '03' AND SUBSTR(time, 12, 5) < '22:01') 
				OR 
				(day_of_week > 4  AND SUBSTR(time, 12, 2) > '07' AND SUBSTR(time, 12, 5) < '22:01');
				"""
	else:
		query = 'SELECT * FROM occupancy'

	return get_data_from_database(query)


def add_lines_info_to_data(input_data_frame):
	"""
	Adds all data from database table `lines_usage` to DataFrame given as input argument.
	DataFrame in input argument should be in format from method
	get_all_occupancy_data. This method will then adds to each
	time sample information about lines usage.
	:param input_data_frame: DataFrame with occupancy data
	:return: DataFrame with occupancy and lines usage info
	"""
	data_frame = input_data_frame.copy()
	df['lines_reserved'].values[:] = 0
	for index, row in input_data_frame.iterrows():
		ts = datetime.datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
		organisations = get_lines_usage_for_time_stamp(ts)
		data_frame = update_organisations(data_frame, index, organisations)
	data_frame['reserved_Unknown'] = 0
	return data_frame


def update_organisations(data_frame, index, organisations):
	"""
	Takes dictionary with names of organisations and number of reserved lines and updates DataFrame with this
	information. If organisation does not have it's own column in DataFrame it is created with zeros filled.
	Than is row at the position of `index` assigned number of reserved lines from `organisations` dictionary in the
	column belonging to the organisation.
	:param data_frame: DataFrame where to add data
	:param index: Row number in data_frame where to add data
	:param organisations: Dictionary where key is the name of organisation and value is number of reserved lines
	:return: Updated data_frame
	"""
	total_lines = 0
	for organisation, number_of_lines in organisations.items():
		total_lines += number_of_lines
		data_frame_column = 'reserved_' + organisation
		if data_frame_column not in data_frame:
			data_frame[data_frame_column] = 0
		data_frame.at[index, data_frame_column] = number_of_lines
	data_frame.at[index, 'lines_reserved'] = total_lines
	return data_frame


def get_lines_usage_for_time_stamp(ts):
	"""
	Returns number of reserved lines and names of organisations that made reservation at given time stamp `ts`.
	:param ts: datetime time stamp
	:return: Dictionary where key is the name of name of organisation that made reservation and value is number
				of reserved lines
	"""
	time_slot = get_time_slot(ts.hour, ts.minute)
	day = ts.strftime('%Y-%m-%d')
	query = 'SELECT reservation FROM lines_usage WHERE DATE(date) = \'%s\' AND time_slot=\'%s\';' % (day, time_slot)
	data = get_data_from_database(query)
	organisations = dict()

	if not data.empty:
		org_string = str(data['reservation'].values[0])[:-1]
		org_list = org_string.split(',')
		for organisation in org_list:
			if organisation not in organisations:
				organisations[organisation] = 0
			organisations[organisation] += 1

	return organisations


def get_weather_for_time_stamp(ts):
	"""
	Returns weather information at given time stamp `ts`.
	:param ts: datetime time stamp
	:return: WeatherData class instance with all weather information at given time stamp from all stations
	"""
	ts_string = ts.strftime('%Y-%m-%d %H:%M:%S')
	weather_data = WeatherData(ts, 1)
	query =	"""SELECT w.time, w.temperature, w.wind, w.humidity, w.precipitation, w.pressure, w.station,
			abs(strftime(\'%%s\', \'%s\') - strftime(\'%%s\', w.time)) as 'closest_time'
			FROM weather_history w ORDER BY  abs(strftime(\'%%s\', \'%s\') - strftime(\'%%s\', time)) 
			limit 30;""" % (ts_string, ts_string)
	data = get_data_from_database(query)
	weather_data.set_data_from_database(data)
	return weather_data


def add_weather_info_to_data(data_frame):
	"""
	Adds all data from database table `weather_history` and
	adds it to DataFrame given as input argument.
	DataFrame in input argument should be in format from method
	get_all_occupancy_data or add_lines_info_to_data. This method
	will then adds to each time sample information about weather.
	:param data_frame: DataFrame with occupancy and/or line usage data
	:return: Input DataFrame with weather history added
	"""
	data_frame['temperature'] = 0
	data_frame['wind'] = 0
	data_frame['humidity'] = 0
	data_frame['precipitation'] = 0
	data_frame['pressure'] = 1013
	for index, row in data_frame.iterrows():
		ts = datetime.datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
		weather_data = get_weather_for_time_stamp(ts)
		data_frame.at[index, 'temperature'] = weather_data.get_temperature()
		data_frame.at[index, 'wind'] = weather_data.get_wind()
		data_frame.at[index, 'humidity'] = weather_data.get_humidity()
		data_frame.at[index, 'precipitation'] = weather_data.get_precipitation()
		data_frame.at[index, 'pressure'] = weather_data.get_pressure()
	return data_frame


def get_time_slot(hour, minute):
	"""
	Computes time_slot id for given hour and minute. Time slot is corresponding to 15 minutes time
	slots in reservation table on Sutka pool web page (https://www.sutka.eu/en/obsazenost-bazenu).
	time_slot = 0 is time from 6:00 to 6:15, time_slot = 59 is time from 20:45 to 21:00.
	:param hour: hour of the day in 24 hour format
	:param minute: minute of the hour
	:return: time slot for line usage
	"""
	slot_id = (hour - 6)*4 + int(minute/15)
	return slot_id


def save_data_to_csv(data_frame, csv_path):
	"""
	Save given DataFrame `df` to csv file `csv_path`
	:param data_frame: DataFrame to be exported to csv
	:param csv_path: Path where to save csv export
	"""
	data_frame.to_csv(csv_path, index=False)


def clean_data(data_frame):
	"""
	Clean table with data from row that are not useful for machine learning algorithms.
	Method removes days when the pool was closed (cleaning, holidays).
	Also removes columns not needed for ML - they are `id`, `percent`, `park` and `reserved_Odstavka`.
	:param data_frame: DataFrame to be cleaned
	:return: Cleaned DataFrame
	"""
	removed_odstavka = 0
	if 'reserved_Odstavka' in data_frame:
		indices_to_remove = list(df[df['reserved_Odstavka'] > 0].index)
		removed_odstavka += len(indices_to_remove)
		data_frame.drop(indices_to_remove, inplace=True)
		data_frame.drop(columns=['reserved_Odstavka'], inplace=True)

	indices_to_remove = list()
	data_frame.drop(columns=['id', 'percent', 'park'],  inplace=True)

	for index, row in data_frame.iterrows():
		if row['pool'] == 0:
			if int(row['day_of_week']) > 4 and 11 < int(row['time'][11:13]) < 20:
				indices_to_remove.append(index)
			elif 6 < int(row['time'][11:13]) < 20:
				indices_to_remove.append(index)

	print('Removing %d rows from dataset.' % (len(indices_to_remove) + removed_odstavka))
	data_frame.drop(indices_to_remove, inplace=True)
	return data_frame


def add_public_holidays(data_frame):
	"""
	Adds new column holidays into `data_frame` with default value 0. Than iterates
	through whole data_frame and add 1 into holidays column to each time stamp that occurs on public holiday in
	Czech Republic. Holiday dates for years 2017, 2018 and 2019 are listed in list below. 
	:param data_frame: DataFrame where to add public holiday info
	:return: Updated data_frame
	"""""
	holidays = [
		'2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-08', '2017-07-05', '2017-07-06',
		'2017-09-28', '2017-10-28', '2017-11-17', '2017-12-24', '2017-12-25', '2017-12-26', '2018-01-01',
		'2018-03-30', '2018-04-02', '2018-05-01', '2018-05-08', '2018-07-05', '2018-07-06', '2018-09-28',
		'2018-10-28', '2018-11-17', '2018-12-24', '2018-12-25', '2018-12-26', '2019-01-01', '2019-04-19',
		'2019-04-22', '2019-05-01', '2019-05-08', '2019-07-05', '2019-07-06', '2019-09-28', '2019-10-28',
		'2019-11-17', '2019-12-24', '2019-12-25', '2019-12-26', '2020-01-01', '2020-04-10', '2020-04-13']

	data_frame['holiday'] = 0
	for index, row in data_frame.iterrows():
		if str(row['time'])[:10] in holidays:
			data_frame.at[index, 'holiday'] = 1

	return data_frame


def resample_timestamp(data_frame):
	"""
	Generates new columns `month`, `day`, `hour` and `minute` from time stamp in format
	'YYYY-MM-DD HH:mm:ss' in column `time`. time column will than not be used for machine learning
	and used only for splitting data to train/validation/test datasets.
	Four new columns will be used for training and prediction.	
	:return: Updated data_frame
	"""
	data_frame['month'] = 0
	data_frame['day'] = 0
	data_frame['hour'] = 0
	data_frame['minute'] = 0

	for index, row in data_frame.iterrows():
		ts = row['time']
		data_frame.at[index, 'month'] = ts[5:7]
		data_frame.at[index, 'day'] = ts[8:10]
		data_frame.at[index, 'hour'] = ts[11:13]
		data_frame.at[index, 'minute'] = ts[14:16]

	return data_frame


def split_csv(data_frame, train_portion=0.4, validation_portion=0.2):
	"""
	Splits data_frame into train, validation and test part and save them into 3 csv files.
	:param data_frame: DataFrame to be split and saved
	:param train_portion: Float from 0.0 to 1.0 representing portion of data used for training.
			train_portion + validation_portion must be less than 1.0
	:param validation_portion: Float from 0.0 to 1.0 representing portion of data used for validation.
	"""
	rows = data_frame.index.values
	Random(RANDOM_SEED).shuffle(rows)

	n_rows = len(rows)
	n_train_data = int(n_rows * train_portion)
	n_validation_data = int(n_rows * validation_portion)

	train_data = data_frame.iloc[rows[1:n_train_data], :]
	validation_data = data_frame.iloc[rows[n_train_data:n_train_data+n_validation_data], :]
	test_data = data_frame.iloc[rows[n_train_data + n_validation_data:], :]

	save_data_to_csv(train_data, TRAIN_DATASET_CSV_PATH)
	save_data_to_csv(validation_data, VALIDATION_DATASET_CSV_PATH)
	save_data_to_csv(test_data, TEST_DATASET_CSV_PATH)


def generate_csv():
	"""
	Generates csv file to configuration.DATASET_CSV_PATH path from
	SQLite database located at configuration.DB_PATH
	"""
	data_frame = get_all_occupancy_data()
	data_frame = resample_timestamp(data_frame)
	data_frame = add_public_holidays(data_frame)
	data_frame = add_weather_info_to_data(data_frame)
	data_frame = add_lines_info_to_data(data_frame)
	data_frame = clean_data(data_frame)
	save_data_to_csv(data_frame, DATASET_CSV_PATH)


def load_csv():
	"""
	Loads generated csv file from configuration.DATASET_CSV_PATH
	:return: DataFrame with loaded csv file. Empty DataFrame if file does not exist.
	"""
	try:
		df = pd.read_csv(DATASET_CSV_PATH)
	except:
		print('Error reading %s. Make sure file exists or try to regenerate it using generate_csv() method.')
		df = pd.DataFrame()

	return df


def scatter_plot_attendance_dependency(column, data, remove_zero_attendance=True):
	"""
	Plots scatter plot where x axis shows attendance and y axis shows values from `column`
	:param column: Name of column to use for plotting
	:param data: DataFrame with all data
	:param remove_zero_attendance: True if only data where pool attendance is above zero are used
	"""
	if remove_zero_attendance:
		plt.scatter(data[data['pool'] > 0][column], data[data['pool'] > 0]['pool'], c="g", alpha=0.01)
	else:
		plt.scatter(data[column], data['pool'], c="g", alpha=0.01)
	plt.ylabel("Attendance")
	plt.xlabel(column)
	plt.show()


def cut_weather(data_frame, drop_original=False):
	"""
	Resample weather data into bins.
	:param data_frame: DataFrame to resample
	:param drop_original: True if original weather columns should be dropped from data_frame
	:return: Resampled data_frame
	"""
	bins_temperature = [-100, -5, 0, 5, 10, 15, 20, 25, 100]
	data_frame['temperature_binned'] = pd.cut(data_frame['temperature'], bins=bins_temperature, labels=False)
	bins_wind = [-1, 1, 5, 10, 15, 20, 1000]
	data_frame['wind_binned'] = pd.cut(data_frame['wind'], bins=bins_wind, labels=False)
	bins_humidity = [-1, 20, 40, 60, 80, 100]
	data_frame['humidity_binned'] = pd.cut(data_frame['humidity'], bins=bins_humidity, labels=False)
	bins_precipitation = [-1.0, 0.1, 5.0, 10.0, 1000.0]
	data_frame['precipitation_binned'] = pd.cut(data_frame['precipitation'], bins=bins_precipitation, labels=False)
	bins_pressure = [0, 1000, 1010, 1020, 1030, 2000]
	data_frame['pressure_binned'] = pd.cut(data_frame['pressure'], bins=bins_pressure, labels=False)

	if drop_original:
		data_frame.drop(columns=['temperature'], inplace=True)
		data_frame.drop(columns=['wind'], inplace=True)
		data_frame.drop(columns=['humidity'], inplace=True)
		data_frame.drop(columns=['precipitation'], inplace=True)
		data_frame.drop(columns=['pressure'], inplace=True)

	return data_frame


def cut_lines_reservation(data_frame, drop_limit=200):
	"""
	Drops reservation columns with less than `drop_limit` time stamps
	in reservation and move their values to new column `reserved_other`.
	:param data_frame: DataFrame to resample
	:param drop_limit: Limit for nonzero rows for drop
	:return: Resampled data_frame
	"""
	to_drop = ['reserved_Unknown']
	df_other = data_frame['reserved_Unknown']
	for column in data_frame.columns:
		if column.startswith('reserved_'):
			total_reservations = sum(data_frame[column] > 0)
			if total_reservations < drop_limit:
				to_drop.append(column)
				df_other = df_other.add(data_frame[column], axis='index')

	data_frame['reserved_other'] = df_other
	data_frame.drop(columns=to_drop, inplace=True)
	return data_frame


if __name__ == '__main__':
	df = load_csv()
	df = cut_weather(df, True)
	df = cut_lines_reservation(df)
	save_data_to_csv(df, DATASET_CSV_PATH)


