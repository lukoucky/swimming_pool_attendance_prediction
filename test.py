from configuration import *
from models.weighted_average import WeightedAverage
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# TODO: Compare algorithms and compute RMSE for 16h prediction, 14h prediction, 12h prediction, ....
# This will give plots with increasing accuracy and nice to compare

if __name__ == '__main__':
	with open('data/days.pickle', 'rb') as input_file:
			d_train, d_val, d_test = pickle.load(input_file)

	df = d_train[0].data.iloc[100:110,0:5]


	df[df['pool'] < 0] = 0
	df[df['pool'] > 400] = 400
	df['pool'] = df['pool']/400
	
	df['lines_reserved'] = df['lines_reserved']/8
	df['day_of_week'] = df['day_of_week']/6
	df['month'] = df['month']/12
	df['day'] = df['day']/31
	df['hour'] = df['hour']/24
	df['minute'] = df['minute']/60

	df[df['temperature'] < -45] = -45
	df[df['temperature'] > 45] = 45
	df['temperature'] = (df['temperature']+45.0)/90.0

	df[df['wind'] < 0] = 0
	df[df['wind'] > 100] = 100
	df['wind'] = df['wind']/100

	df[df['humidity'] < 0] = 0
	df[df['humidity'] > 100] = 100
	df['humidity'] = df['humidity']/100

	df[df['precipitation'] < 0] = 0
	df[df['precipitation'] > 100] = 100
	df['precipitation'] = df['precipitation']/100

	df[df['pressure'] < 800] = 800
	df[df['pressure'] > 1200] = 1200
	df['pressure'] = (df['pressure']-800)/400

	print(df)
