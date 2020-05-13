from models.model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import pickle
import os

class RandomForest(Model):

	def __init__(self):
		super().__init__(RandomForestClassifier(n_estimators=20, max_features=20, max_depth=50, min_samples_split=5, min_samples_leaf=1))

	def tune_parameters(self, data_x, data_y):
		# bootstrap = [True, False]
		# max_depth = [10, 50, 100, None] 
		# max_features = [20, 50, 'auto', None] 
		# min_samples_leaf = [1, 2, 4]
		# min_samples_split = [2, 5, 10] 
		# n_estimators = [30, 40, 50] 

		max_depth = [10, 50] 
		max_features = [4, 10, None] 
		min_samples_leaf = [1, 2, 4]
		min_samples_split = [2, 5, 10] 
		n_estimators = [20] 

		random_grid = {	'n_estimators': n_estimators,
						'max_features': max_features,
						'max_depth': max_depth,
						'min_samples_split': min_samples_split,
						'min_samples_leaf': min_samples_leaf}

		self.random_param_search = RandomizedSearchCV(estimator=self.model, param_distributions=random_grid, n_iter=2, cv=3, verbose=0, random_state=17, n_jobs=4, scoring=make_scorer(mean_squared_error,  greater_is_better=False))
		self.random_param_search.fit(data_x, data_y)
		self.are_parameters_tuned = True
		print('Best parameters for RandomForest:')
		print(self.random_param_search.best_params_)

		with open('data/RandomForestClassifier_RandomSearch.pickle', 'wb') as f:
			pickle.dump(self.random_param_search, f)

	def fit(self, without_reserved=False):
		x_train, y_train, x_test, y_test = self.load_data(without_reserved)
		self.model.fit(x_train, y_train)
		self.are_parameters_tuned = True

	def predict(self, data):
		pass

	def load_tuned_parameters(self, pickle_path=None):
		if pickle_path is None:
			pickle_path = 'data/RandomForestClassifier_RandomSearch.pickle'
		with open(pickle_path, 'rb') as f:
			self.random_param_search = pickle.load(f)
		self.are_parameters_tuned = True

	def load_model(self, pickle_path='data/rfc.pickle'):
		with open(pickle_path, 'rb') as f:
			self.model = pickle.load(f)
		self.are_parameters_tuned = True

	def save_model(self, pickle_path='data/rfc.pickle'):
		with open(pickle_path, 'wb') as f:
			pickle.dump(self.model, f)
