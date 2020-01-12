from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from data_helper import DataHelper
import numpy as np
import pickle


class TreeModelBase():
	"""
	Base class for all models based on Decision trees like Random Forest or Extra Trees
	"""
	def __init__(self):
		"""
		Constructor
		"""
		self.dh = DataHelper()

	def load_model(self, pickle_path=None):
		"""
		Loads model from pickle in pickle_path. If pickle_path is None loads from default location defined by self.name
		:param pickle_path: path where to load model from
		"""
		if pickle_path is None:
			pickle_path = 'data/'+self.name+'.pickle'

		with open(pickle_path, 'rb') as f:
			self.model = pickle.load(f)

	def save_model(self, pickle_path=None):
		"""
		Saves model to pickle in pickle_path. If pickle_path is None saves to default location defined by self.name
		:param pickle_path: path where to save model with file name
		"""
		if pickle_path is None:
			pickle_path = 'data/'+self.name+'.pickle'

		with open(pickle_path, 'wb') as f:
			pickle.dump(self.model, f)

	def fit_on_training_set(self):
		"""
		Fit model on training data.
		"""
		x_train, y_train, x_test, y_test = self.dh.generate_feature_vectors(self.columns, self.time_steps_back)
		self.model.fit(x_train, y_train.ravel())

	def get_mse(self):
		"""
		Computes mean square error on model
		:return: mean square error
		"""
		return self.dh.mse_on_testing_days(self.model, self.columns, self.time_steps_back)

	def show_n_predictions(self, n):
		"""
		Plots `n` predictions from training data using this model.
		:param n: If integer then represents number of random testing days. If list of integers
					then represents day ids from testing days. Last possible option is 
					string `all` that will plot all testing days.
		"""
		self.dh.show_n_days_prediction(self.model, self.columns, n, self.time_steps_back)

	def print_mse(self):
		"""
		Prints mean square error
		"""
		print('\nMSE = %.2f\n'%(self.get_mse()))


class MyExtraTreesRegressor(TreeModelBase):

	def __init__(self):
		super(MyExtraTreesRegressor, self).__init__()
		self.model = ExtraTreesRegressor(random_state=17, n_estimators=65, max_depth=35, min_samples_split=2, min_samples_leaf=1, max_features=40, max_leaf_nodes=None)
		self.time_steps_back = 9
		self.columns = ['pool', 'lines_reserved', 'day_of_week', 'month', 'day', 'hour', 'minute', 'holiday', 'reserved_Lavoda', 'reserved_Club Junior', 'reserved_Elab', 'reserved_Vodnik', 'reserved_Spirala', 'reserved_Amalka', 'reserved_Dukla', 'reserved_Lodicka', 'reserved_Elab team', 'reserved_Sports Team', 'reserved_Modra Hvezda', 'reserved_VSC MSMT', 'reserved_Orka', 'reserved_Activity', 'reserved_Aquamen', 'reserved_Zralok', 'reserved_SK Impuls', 'reserved_Motylek', 'reserved_3fit', 'reserved_Jitka Vachtova', 'reserved_Hodbod', 'reserved_DUFA', 'reserved_The Swim', 'reserved_Neptun', 'reserved_Strahov Cup', 'reserved_Apneaman', 'reserved_Michovsky', 'reserved_Betri', 'reserved_Pospisil', 'reserved_Vachtova', 'reserved_Riverside', 'reserved_Vodni polo Sparta', 'reserved_Road 2 Kona', 'reserved_Water Polo Sparta Praha', 'reserved_Sucha', 'reserved_Totkovicova', 'reserved_DDM Spirala', 'reserved_PS Perla', 'reserved_Dufkova - pulka drahy', 'reserved_Pavlovec', 'reserved_Sidorovich', 'reserved_OS DUFA', 'temperature_binned', 'wind_binned', 'humidity_binned', 'precipitation_binned', 'pressure_binned', 'reserved_other', 'minute_of_day', 'year']
		self.name = 'MyExtraTreesRegressor'


class MyExtraTreesClassifier(TreeModelBase):

	def __init__(self):
		super(MyExtraTreesClassifier, self).__init__()
		self.model = ExtraTreesClassifier(random_state=17, n_estimators=10, max_depth=50, min_samples_split=5, min_samples_leaf=2)
		self.time_steps_back = 10
		self.columns = ['pool','day_of_week','month','minute_of_day', 'year', 'reserved_Vodnik','lines_reserved']
		self.name = 'MyExtraTreesClassifier'

class MyRandomForestRegressor(TreeModelBase):

	def __init__(self):
		super(MyRandomForestRegressor, self).__init__()
		self.model = RandomForestRegressor(random_state=17, n_estimators=30, max_depth=20, min_samples_split=5, min_samples_leaf=1, max_features=20, max_leaf_nodes=None)
		self.time_steps_back = 10
		self.columns = ['pool', 'lines_reserved', 'day_of_week', 'month', 'day', 'hour', 'minute', 'holiday', 'reserved_Lavoda', 'reserved_Club Junior', 'reserved_Elab', 'reserved_Vodnik', 'reserved_Spirala', 'reserved_Amalka', 'reserved_Dukla', 'reserved_Lodicka', 'reserved_Elab team', 'reserved_Sports Team', 'reserved_Modra Hvezda', 'reserved_VSC MSMT', 'reserved_Orka', 'reserved_Activity', 'reserved_Aquamen', 'reserved_Zralok', 'reserved_SK Impuls', 'reserved_Motylek', 'reserved_3fit', 'reserved_Jitka Vachtova', 'reserved_Hodbod', 'reserved_DUFA', 'reserved_The Swim', 'reserved_Neptun', 'reserved_Strahov Cup', 'reserved_Apneaman', 'reserved_Michovsky', 'reserved_Betri', 'reserved_Pospisil', 'reserved_Vachtova', 'reserved_Riverside', 'reserved_Vodni polo Sparta', 'reserved_Road 2 Kona', 'reserved_Water Polo Sparta Praha', 'reserved_Sucha', 'reserved_Totkovicova', 'reserved_DDM Spirala', 'reserved_PS Perla', 'reserved_Dufkova - pulka drahy', 'reserved_Pavlovec', 'reserved_Sidorovich', 'reserved_OS DUFA', 'temperature_binned', 'wind_binned', 'humidity_binned', 'precipitation_binned', 'pressure_binned', 'reserved_other', 'minute_of_day', 'year']
		self.name = 'MyRandomForestRegressor'


class MyRandomForestClassifier(TreeModelBase):

	def __init__(self):
		super(MyRandomForestClassifier, self).__init__()
		self.model = RandomForestClassifier(random_state=17, n_estimators=10, max_depth=30, min_samples_split=2, min_samples_leaf=2)
		self.time_steps_back = 5
		self.columns = ['pool','day_of_week','month','minute_of_day', 'year', 'reserved_Vodnik','lines_reserved']
		self.name = 'MyRandomForestClassifier'
