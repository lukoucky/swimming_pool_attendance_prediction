from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json

class LSTM_classifier():

	def __init__(self):
		self.train_x = None
		self.test_x = None
		self.validation_x = None
		self.train_y = None
		self.test_y = None
		self.validation_y = None
		self.data_set = False
		self.model = Sequential()


	def add_data(self, train_x, validation_x, test_x, train_y, validation_y, test_y):
		self.train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
		self.test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
		self.validation_x = validation_x.reshape((validation_x.shape[0], 1, validation_x.shape[1]))
		self.train_y = train_y
		self.test_y = test_y
		self.validation_y = validation_y
		print('LSTM data added. Shapes:', self.train_x.shape, self.train_y.shape, self.validation_x.shape, self.validation_y.shape, self.test_x.shape, self.test_y.shape)
		self.data_set = True


	def train_model(self):
		self.model.add(LSTM(50, input_shape=(self.train_x.shape[1], self.train_x.shape[2])))
		self.model.add(Dense(1))
		self.model.compile(loss='mae', optimizer='adam')

		history = self.model.fit(self.train_x, self.train_y, epochs=50, batch_size=6, validation_data=(self.validation_x, self.validation_y), verbose=2, shuffle=False)

		pyplot.plot(history.history['loss'], label='train')
		pyplot.plot(history.history['val_loss'], label='test')
		pyplot.legend()
		pyplot.show()

	def predict_test(self):
		yhat = self.model.predict(self.test_x)
		rmse = sqrt(mean_squared_error(self.test_y, yhat))
		print('Test RMSE: %.3f' % rmse)

	def save_model(self):
		model_json = self.model.to_json()
		with open('model.json', 'w') as json_file:
			json_file.write(model_json)
		self.model.save_weights("model.h5")
		print("Saved model to disk")

	def load_model(self):
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights("model.h5")
		print("Loaded model from disk")

	def predict_day(self, day, without_reserved=False):
		if without_reserved:
			x, y = day.build_timeseries_without_reservations()
		else:
			x, y = day.build_timeseries(3, True)
		y_pred = list()
		for i, data in enumerate(x):
			if i > 3:
				data[0] = y_pred[-3]
				data[int(x.shape[1]/3)] = y_pred[-2]
				data[int((x.shape[1]/3)*2)] = y_pred[-1]
			data_shaped = data.reshape((1, 1, data.shape[0]))
			prediction = self.model.predict(data_shaped)
			y_pred.append(prediction[0][0]*400)
		return y_pred