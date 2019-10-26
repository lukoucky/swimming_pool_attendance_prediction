from models.random_forest_classifier import RandomForest
from utils import MyGridSearch
from utils import Helper
from sklearn.ensemble import RandomForestClassifier

# rf = RandomForest()
# # rf.tune_parameters_on_full_training_data()
# rf.load_tuned_parameters()
# rf.set_best_tuned_parameters()
# rf.evaluate_model_on_testing_data()
# # rf.show_n_days_prediction(1)
# rf.get_testing_mse()

bootstrap = [True, False]
max_depth = [10, 50, 100, None] 
max_features = [20, 50, None] 
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 5, 10] 
n_estimators = [10, 20] 
random_grid = {	'n_estimators': n_estimators,
				'max_features': max_features,
				'max_depth': max_depth,
				'min_samples_split': min_samples_split,
				'min_samples_leaf': min_samples_leaf}

mgs = MyGridSearch('RandomForestClassifier',random_grid)
mgs.fit()