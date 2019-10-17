import pandas as pd
import numpy as np
from configuration import *
from models.weighted_average import WeightedAverage

def trim_dataset(data, batch_size):
	"""
	Trims dataset to a size that's divisible by BATCH_SIZE
	:param data: D
	:param batch_size:
	:return:
	"""
	no_of_rows_drop = data.shape[0]%batch_size
	if(no_of_rows_drop > 0):
		return data[:-no_of_rows_drop]
	else:
		return data

def build_timeseries(mat, y_col_index):
	"""

	"""
	# y_col_index is the index of column that would act as output column
	# total number of time-series samples would be len(mat) - TIME_STEPS
	TIME_STEPS = 3
	dim_0 = mat.shape[0] - TIME_STEPS
	dim_1 = mat.shape[1]
	x = np.zeros((dim_0, TIME_STEPS, dim_1))
	y = np.zeros((dim_0,))
	
	for i in range(dim_0):
		print(i)
		x[i] = mat[i:TIME_STEPS+i]
		y[i] = mat[TIME_STEPS+i, y_col_index]
	print("length of time-series i/o",x.shape,y.shape)
	return x, y


if __name__ == '__main__':
	train_data = pd.read_csv(TRAIN_DATASET_CSV_PATH)
	train_data = train_data.iloc[1:10]

	data = train_data.to_numpy()
	print(data)
	x, y = build_timeseries(data, 0)
	print(x)
	print(y)
