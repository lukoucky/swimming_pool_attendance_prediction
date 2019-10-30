from models.random_forest_classifier import RandomForest
from utils import MyGridSearch
from sklearn.ensemble import RandomForestClassifier
from utils import Day, DaysStatistics
import matplotlib.pyplot as plt


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
	d_train, d_val, d_test = rf.get_all_days_lists()
	x,y = rf.build_feature_vector_with_average(d_train+d_val)
	# rf.without_reserved = True
	rf.model.fit(x, y)
	rf.are_parameters_tuned = True
	rf.save_model('data/rfc_with_avg.pickle')
	rf.show_n_days_prediction(12)
	# print(rf.get_testing_mse())

	# d_train, d_val, d_test = rf.get_all_days_lists()
	# ds = DaysStatistics(d_train + d_val + d_test)

	# day_id = 63
	# d = list(d_train[day_id].data['pool'])
	# month = d_train[day_id].data.iloc[0]['month']
	# day_of_week = d_train[day_id].data.iloc[0]['day_of_week']
	# weekend = False
	# offset = 48
	# if day_of_week > 4:
	# 	weekend = True
	# 	offset = 96
	# ds.plot_monthly_average(month-1, weekend, d, offset)

	# ds.plot_year_averages_by_month(True)
	# ds.plot_year_averages_by_month(False)
