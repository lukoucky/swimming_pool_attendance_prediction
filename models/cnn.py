from models.neural_network_base import NeuralNetworkBase
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


class ConvolutionalNeuralNetwork(NeuralNetworkBase):
	"""
	Convolutional Neural Network model. Using 1D convolutional layers at input.
	"""
	def __init__(self, model_name=None):
		"""
		Constructor initializies base constructor and sets default model_name if not given in input argument
		"""
		super(ConvolutionalNeuralNetwork, self).__init__(model_name)
		if self.model_name is None:
			self.model_name = 'cnn_model'		
		self.build_model()

	def build_model(self):
		"""
		Builds and compiles CNN model.
		"""
		n_features = len(self.columns)
		self.model = Sequential()
		self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.time_steps_back, n_features)))
		self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Flatten())
		self.model.add(Dense(100, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(optimizer='adam', loss='mse')
