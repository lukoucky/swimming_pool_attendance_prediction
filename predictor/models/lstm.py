from models.neural_network_base import NeuralNetworkBase
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


class LongShortTermMemory(NeuralNetworkBase):
	"""
	Long Short Term Memory model. Using LSTM layers.
	"""
	def __init__(self, model_name=None, units=150, model_type='bidirectional'):
		"""
		Constructor initializies base constructor and sets default model_name if not given in input argument
		:param model_name: Name of model used for saving and loading
		:param units: Number of LSTM units in each layer
		:param model_type: Type of model. `bidirectional` will result in Bidirectional LSTM model. Other with normal LSTM model
		"""
		super(LongShortTermMemory, self).__init__(model_name)
		if self.model_name is None:
			self.model_name = 'lstm_model'		

		self.units = units

		if model_type.startswith('bidir'):
			self.model_type = 'bidirectional'
			self.build_bidirectional_model(units)
		else:
			self.model_type = 'normal'
			self.build_model(units)

	def build_model(self, units=150):
		"""
		Builds and compiles LSTM model.
		:param units: Number of LSTM units
		"""
		n_features = len(self.columns)
		self.model = Sequential()
		self.model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=(self.time_steps_back, n_features)))
		self.model.add(LSTM(units, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer='adam', loss='mse')

	def build_bidirectional_model(self, units=150):
		"""
		Builds and compiles bidirectional LSTM model.
		:param units: Number of LSTM units
		"""
		n_features = len(self.columns)
		self.model = Sequential()
		self.model.add(Bidirectional(LSTM(units, activation='relu', return_sequences=True, input_shape=(self.time_steps_back, n_features))))
		self.model.add(Bidirectional(LSTM(units, activation='relu')))
		self.model.add(Dense(1))
		self.model.compile(optimizer='adam', loss='mse')