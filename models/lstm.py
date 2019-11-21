from models.neural_network_base import NeuralNetworkBase
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


class LongShortTermMemory(NeuralNetworkBase):
	"""
	Long Short Term Memory model. Using LSTM layers.
	"""
	def __init__(self, model_name=None):
		"""
		Constructor initializies base constructor and sets default model_name if not given in input argument
		"""
		super(LongShortTermMemory, self).__init__(model_name)
		if self.model_name is None:
			self.model_name = 'lstm_model'		
		self.build_model()

	def build_model(self):
		"""
		Builds and compiles LSTM model.
		"""
		n_features = len(self.columns)
		self.model = Sequential()
		self.model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(self.time_steps_back, n_features)))
		self.model.add(LSTM(100, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer='adam', loss='mse')
