from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import pickle

class RandomForest():

	def __init__(self):
		self.model = ExtraTreesClassifier()
		self.random_param_search = None

	def tune_parameters(self, data_x, data_y):
		bootstrap = [True, False]
		max_depth = [10, 50, 100, None] 
		max_features = [20, 50, 'auto', None] 
		min_samples_leaf = [1, 2, 4]
		min_samples_split = [2, 5, 10] 
		n_estimators = [30, 40, 50] 

		random_grid = {	'n_estimators': n_estimators,
						'max_features': max_features,
						'max_depth': max_depth,
						'min_samples_split': min_samples_split,
						'min_samples_leaf': min_samples_leaf,
						'bootstrap': bootstrap}

		mse_scorer = make_scorer(mean_squared_error,  greater_is_better=False)

		self.random_param_search = RandomizedSearchCV(	estimator=self.model, 
														param_distributions=random_grid, 
														n_iter=100, 
														cv=3, 
														verbose=2, 
														random_state=17, 
														n_jobs=4, 
														scoring=mse_scorer)
		self.random_param_search.fit(data_x, data_y)
		print('Best parameters:', self.random_param_search.best_params_)

		with open('data/extra_param_search.pickle', 'wb') as f:
			pickle.dump(self.random_param_search, f)



	def evaluate(self, data_x, data_y):
		best_rfc = self.random_param_search.best_estimator_
		pred_y = best_rfc.predict(data_x)
		print('mean_squared_error', mean_squared_error(pred_y, data_y))

	def fit(self, data):
		pass

	def predict(self, data):
		pass

if __name__=='__main__':
	with open('data/data.pickle', 'rb') as f:
		x_train, y_train, x_test, y_test = pickle.load(f)
	rf = RandomForest()
	rf.tune_parameters(x_train, y_train)
	rf.evaluate(x_test, y_test)
