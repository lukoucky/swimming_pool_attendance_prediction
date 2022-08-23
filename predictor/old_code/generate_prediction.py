from models.tree_models import DoubleExtraTreesRegressor
from datetime import timedelta, date, datetime
from predictor import Predictor
import os

# TODO: change to evnironemt variables - also the number of days
path_to_model = '/home/models/DoubleExtraTreesRegressor.pickle'
export_folder = '/var/www/html/data/prediction_extra_tree'

def double_extra_trees(n_days=5):
    """
    Generates prediction for today and n_days-1 days in to the future
    using DoubleExtraTreesRegressor
    """
    p = Predictor()
    d = datetime.now()
    clf = DoubleExtraTreesRegressor()

    if os.path.isfile(path_to_model):
        clf.load_model(path_to_model)
    else:
        clf.fit_on_training_set()
        clf.save_model(path_to_model)

    for i in range(n_days):
        prediction_date = datetime.now() + timedelta(days=i)
        p.generate_predictions(clf, prediction_date, export_folder)


if __name__ == '__main__':
    double_extra_trees(5)