import inspect
import itertools
import pickle
from data_helper import DataHelper as DataHelper

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class MyGridSearch:
    """
    Naive implementation of grid seach for hyperparameter tuning.
    Motivation for this class is the fact that sklearn grid seach does not let you define 
    evaluation method (or at least I did not find how ot do it). This class is using evaluation
    from DataHelper using method `mse_on_testing_days`. It computes mean squared error on all testing 
    days for given predictor. The importatnt part is that this method starts predicting at the begining 
    of the day and than use prediction outputs as inputs for the next time steps. That way is simulated
    real usage of predictor. Since this implementation uses eval() method on strings all the 
    supported sklearn classifiers must be imported here before grid search starts.
    """
    def __init__(self, estimator_name, param_dict):
        """
        Constructor of MyGridSearch. Prepares all necessary variables and prepares generator
        for grid search.
        :param estimator_name: String with name of sklearn predictor. Must be exactly the same as 
                                sklearn name. This is because this string is used for creation
                                of predictor. Not the best solution but working one.
        :param param_dict: Disctionary with parameters to search through. Key must be the exact name 
                            of parameter input for predictor, value must be list of sutable values.
                            Current implementation have problems with strings so only numeric values are
                            supported now.
        """
        inspector = inspect.getfullargspec(eval(estimator_name))
        for key in param_dict.keys():
            assert key in inspector.args, 'Argument %s is not valid for class %s' % (key, estimator.__class__.__name__)

        self.estimator_name = estimator_name
        self.param_dict = param_dict
        self.evaluation = []
        self.parameters = []
        self.best_parameters = None
        self.best_evaluation = None
        self.best_estimator = None
        self.arguments = None
        self.generator = None
        self.grid_size = 1
        self.dh = DataHelper()
        self.prepare_generator()
        
    def fit(self, columns_to_keep=None, time_step_back=3):
        """
        Perfoms grid search and saves best predictor into pickle.
        :param columns: List of columns that should remain in generated features. Deafult is None, when 
                        all columns appart from `time` remains. List of columns must be the same as was
                        for data used to fit the `predictor`
        :param time_step_back: Number of time stamps in history that are packed together as input features
        """
        n = 1
        x_train, y_train, x_test, y_test = self.dh.generate_feature_vectors(columns_to_keep, time_step_back)
        for values in self.generator:
            params = {}
            estimator_str = self.estimator_name + '(random_state=17, '
            for i in range(len(self.arguments)):
                estimator_str += self.arguments[i] + '=' + str(values[i]) + ', '
                params[self.arguments[i]] = values[i]
            estimator_str = estimator_str[:-2] + ')'

            self.parameters.append(params)
            e = eval(estimator_str)
            e.fit(x_train, y_train.ravel())
            score = self.dh.mse_on_testing_days(e, columns_to_keep, time_step_back)

            if len(self.evaluation) == 0 or score < min(self.evaluation):
                self.best_evaluation = score
                self.best_estimator = e
                self.best_parameters = params

            self.evaluation.append(score)
            print('%d out of %d done for parameters %s with score %.1f, best MSE so far = %.1f' % (n, self.grid_size, estimator_str, score, self.best_evaluation))
            n +=1 
        print('GridSearch for %s done.\nBest MSE = %.1f for parameters:' % (self.estimator_name, self.best_evaluation))
        print(self.best_parameters)
        print('Saving best estimator')
        est_path = 'data/%s_MyGridSearch_MSE_%.1f.pickle' % (self.estimator_name, self.best_evaluation)
        with open(est_path, 'wb') as f:
            pickle.dump(self.best_estimator, f)

    def prepare_generator(self):
        """
        Prepares all possible combinations of parameters for grid search.
        """
        self.arguments = list()
        product_str = 'itertools.product(['
        for key, value in self.param_dict.items():
            self.grid_size *= len(value)
            self.arguments.append(key)
            for element in value:
                product_str += str(element) + ', '
            product_str = product_str[:-2] + '], ['
        product_str = product_str[:-3] + ')'
        self.generator = eval(product_str)
