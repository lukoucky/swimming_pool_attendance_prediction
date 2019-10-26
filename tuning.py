from models.random_forest_classifier import RandomForest
from utils import MyGridSearch
from sklearn.ensemble import RandomForestClassifier
from utils import Day

def tune_random_forest():
	max_depth = [10, 50] 
	max_features = [2, 5, 10, 20] 
	min_samples_leaf = [1, 2, 4]
	min_samples_split = [2, 5] 
	n_estimators = [10, 20] 
	random_grid = {	'n_estimators': n_estimators,
					'max_features': max_features,
					'max_depth': max_depth,
					'min_samples_split': min_samples_split,
					'min_samples_leaf': min_samples_leaf}

	mgs = MyGridSearch('RandomForestClassifier',random_grid)
	mgs.fit(True)

if __name__ == '__main__':
	rf = RandomForest()
	rf.without_reserved = True
	# rf.fit()
	# rf.save_model('data/rfc_nores.pickle')
	# rf.show_n_days_prediction(12)
	# print(rf.get_testing_mse())

	d_train, d_val, d_test = rf.get_all_days_lists()
	x,y = d_test[20].build_timeseries(8)
	print(x[100], y[100])