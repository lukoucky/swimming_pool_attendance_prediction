import pickle
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class DataHelper():
    """
    Helper class for working with data. This class expects already created lists with
    testing, validation and training data in `data/days.pickle` path. 
    Provides getters for data, generators for features and plotting functions for tuning.
    """
    def __init__(self, csv_path='data/dataset.csv', days_data_path='data/days.pickle'):
        """
        DataHelper constructor.
        :param csv_path: Path to csv with all data exported from SQLite database
        :param days_data_path: Path to pickle with lists conaining instances of Day class with
                                testing, validation and training data.
        """
        self.csv_path = csv_path
        self.days_data_path = days_data_path
        self.days_train = None
        self.days_valid = None
        self.days_test = None
        self.columns = list()
        self.prepare_days_data()
    
    def prepare_days_data(self):
        """
        Loads pickle with all Days
        """
        # TODO: generate days if pickle missing
        with open(self.days_data_path, 'rb') as input_file:
            self.days_train, self.days_valid, self.days_test = pickle.load(input_file)
        self.columns = self.days_train[0].data.columns
        
    def get_all_days_list(self):
        """
        Returns all Day
        :return: List of all Days
        """
        return self.days_train+self.days_valid+self.days_test
    
    def get_testing_days(self):
        """
        Returns Day for testing
        :return: List of Days for testing
        """
        return self.days_test
    
    def get_training_days(self, include_validation=True):
        """
        Returns Day for training
        :param include_validation: If True returns validation and trainig days together, if False only training days
        :return: List of Days for training
        """
        if include_validation:
            return self.days_train + self.days_valid
        else:
            return self.days_train
    
    def get_validation_days(self):
        """
        Returns Day for validation
        :return: List of Days for validation
        """
        return self.days_valid
    
    def get_all_columns_names(self):
        """
        Returns names of all columns in DataFrame in Day
        :return: List with name of all columns in DataFrame in Day
        """
        return self.columns
    
    def reformat_feature_list(self, x, y):
        """
        Ferofmats feature vector and vector with results from list of numpy arrays to big numpy.
        :param x: List of numpy arrays with features
        :param y: List of numpy arrays with results
        :return: Two numpy arrays with features and results
        """
        for i, data in enumerate(x):
            if i > 0:
                x_data = np.concatenate([x_data,data])
            else:
                x_data = np.array(data)

        for i, data in enumerate(y):
            if i > 0:
                y_data = np.concatenate([y_data,data])
            else:
                y_data = np.array(data)
        return x_data, y_data.ravel()
    
    def get_feature_vectors_from_days(self, days_list, columns_to_drop, time_steps_back=3, time_steps_forward=1):
        """
        Creates feature vector and vector with results from Day object.
        :param days_list: List of Days from which are features generated
        :param columns_to_drop: List of columns that should be droped from DataFrame in Day.
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :param time_steps_forward: Number of time stamps in future that are packed together as output results
        :return: Two numpy arrays with features and results
        """
        x, y = list(), list()
        for day in days_list:
            clean_data = day.data.copy()
            clean_data.drop(columns=columns_to_drop, inplace=True)
            x_day, y_day = self.build_timeseries(clean_data, time_steps_back, time_steps_forward)
            x.append(x_day)
            y.append(y_day)
        return self.reformat_feature_list(x, y)
    
    def get_test_day_feature_vectors(self, day_id, columns_to_keep=None, time_steps_back=3, time_steps_forward=1):
        """
        Generates feature and results vector for one testing Day.
        :param day_id: Id of Day from list of testing days
        :param columns_to_keep: List of columns that should remain in generated features. Deafult is None, when 
                                all columns appart from `time` remains.
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :param time_steps_forward: Number of time stamps in future that are packed together as output results
        :return: Two numpy arrays with features and results
        """
        columns_to_drop = list()
        if columns_to_keep is not None:
            for column in self.get_all_columns_names():
                if column not in columns_to_keep:
                    columns_to_drop.append(column)
        else:
            columns_to_drop = ['time']
        
        n_testing_days = len(self.get_testing_days())
        if day_id < n_testing_days:
            x_test, y_test = self.get_feature_vectors_from_days([self.get_testing_days()[day_id]], columns_to_drop, time_steps_back, time_steps_forward)
            return x_test, y_test 
        else:
            print('get_test_day_feature_vectors - requested day ID %d that is out of bounds. There are %d testing days.' % (day_id, n_testing_days))
            return None, None

    
    def generate_feature_vectors(self, columns_to_keep=None, time_steps_back=3, time_steps_forward=1):
        """
        Creates training and testing feature vectors and vectors with results.
        :param columns_to_keep: List of columns that should remain in generated features. Deafult is None, when 
                                all columns appart from `time` remains.
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :param time_steps_forward: Number of time stamps in future that are packed together as output results
        :return: Four numpy arrays with structure: train_features, train_results, test_features, test_results
        """
        columns_to_drop = list()
        if columns_to_keep is not None:
            for column in self.get_all_columns_names():
                if column not in columns_to_keep:
                    columns_to_drop.append(column)
        else:
            columns_to_drop = ['time']
        
        x_train, y_train = self.get_feature_vectors_from_days(self.get_training_days(True), columns_to_drop, time_steps_back, time_steps_forward)
        x_test, y_test = self.get_feature_vectors_from_days(self.get_testing_days(), columns_to_drop, time_steps_back, time_steps_forward)

        return x_train, y_train, x_test, y_test
    
    def generate_feature_vectors_for_hmm(self, columns_to_keep=None):
        """
        Creates training and testing feature vectors and vectors with results.
        :param columns_to_keep: List of columns that should remain in generated features. Deafult is None, when 
                                all columns appart from `time` remains.
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :param time_steps_forward: Number of time stamps in future that are packed together as output results
        :return: Four numpy arrays with structure: train_features, train_results, test_features, test_results
        """
        columns_to_drop = list()
        if columns_to_keep is not None:
            for column in self.get_all_columns_names():
                if column not in columns_to_keep:
                    columns_to_drop.append(column)
        else:
            columns_to_drop = ['time']
        
        x_lengths = []
        for i, day in enumerate(self.get_training_days(True)):
            if i  == 0:
                x_train, y_train = self.get_feature_vectors_from_days([day], columns_to_drop, 1, 1)
                x_lengths.append(len(x_train))
            else:
                new_x, new_y = self.get_feature_vectors_from_days([day], columns_to_drop, 1, 1)
                x_train = np.concatenate([x_train, new_x])
                x_lengths.append(len(new_x))
        # x_test, y_test = self.get_feature_vectors_from_days(self.get_testing_days(), columns_to_drop, 1, 1)

        return x_train, x_lengths

    def build_timeseries(self, data_frame, time_steps_back=3, time_steps_forward=1):
        """
        Creates vector of prediction results and features.
        :param data_frame: DataFrame to build from
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :param time_steps_forward: Number of time stamps in future that are packed together as output results
        :return: Two numpy arrays with features and results
        """
        matrix = data_frame.values
        dim_0 = matrix.shape[0] - time_steps_back
        dim_1 = matrix.shape[1]

        x = np.zeros((dim_0, time_steps_back*dim_1))
        y = np.zeros((dim_0, time_steps_forward))

        for i in range(dim_0):
            x_data = matrix[i:time_steps_back+i]
            x[i] = np.reshape(x_data, x_data.shape[0]*x_data.shape[1])

            end_id = time_steps_back+i+time_steps_forward
            if end_id >= dim_0:
                end_id = dim_0 - 1

            y_data = matrix[time_steps_back+i:end_id, 0]
            if len(y_data) < time_steps_forward:
                y_provis = np.zeros(time_steps_forward,)
                for i, value in enumerate(y_data):
                    y_provis[i] = value
                y[i] = y_provis
            else:
                y[i] = np.reshape(y_data, time_steps_forward)

        return x, y
    
    def predict_day_from_features(self, x, predictor, time_steps_back=3):
        """
        Predicts whole day attandance from feature `x` using `predictor`
        Each prediction step is using results from last predictions.
        :param x: numpy array with input features
        :param predictor: sklearn predictor that have function predict accepting one line from `x` as input
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :return: numpy array predictions
        """
        y_pred = list()
        prediction_ids = [0]*time_steps_back
        for i in range(1,time_steps_back):
            prediction_ids[i] = int(x.shape[1]/time_steps_back)*i
        
        for i, data in enumerate(x):
            if i > time_steps_back:
                y_pred_id = -time_steps_back
                for data_id in prediction_ids:
                    data[data_id] = y_pred[y_pred_id]
                    y_pred_id += 1
            prediction = predictor.predict([data])
            y_pred.append(prediction[0])
        return y_pred
    
    def predict_day(self, day, predictor, time_steps_back=3):
        """
        Predicts whole day attandance for `day` using `predictor`
        Each prediction step is using results from last predictions.
        :param day: Instance of Day object
        :param predictor: sklearn predictor that have function predict accepting one line from `x` as input
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :return: numpy array predictions
        """
        x, y = self.build_timeseries(day.data, time_steps_back)
        return self.predict_day_from_features(x, predictor, time_steps_back)
        
    def mse_on_testing_days(self, predictor, columns= None, time_steps_back=3):
        """
        Computes mean squared error on all testing days for given predictor.
        :param predictor: fitted sklearn predictor
        :param columns: List of columns that should remain in generated features. Deafult is None, when 
                        all columns appart from `time` remains. List of columns must be the same as was
                        for data used to fit the `predictor`
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :return: mean squared error for all testing days
        """
        errors = list()
        n_days = len(self.get_testing_days())
        for i in range(n_days):
            x_day, y_day = self.get_test_day_feature_vectors(i, columns, time_steps_back)
            errors.append(self.mse_on_day(x_day, y_day, predictor, time_steps_back))
        return sum(errors)/n_days

    def mse_on_day(self, x, y, predictor, time_steps_back=3):
        """
        Computes mean squared error for one day for given predictor.
        :param x: Numpy arrays with features
        :param y: Numpy arrays with results
        :param predictor: fitted sklearn predictor
        :param time_step_back: Number of time stamps in history that are packed together as input features
        :return: mean squared error for all testing days
        """
        y_pred = self.predict_day_from_features(x, predictor, time_steps_back)
        return mean_squared_error(y, y_pred)
    
    def show_prediction(self, test_day_id, predictor, columns=None):
        """
        Shows plot for one testing day on index `test_day_id` in list of testing days for given predictor.
        :param test_day_id: index in list of testing days
        :param predictor: fitted sklearn predictor
        :param columns: List of columns that should remain in generated features. Deafult is None, when 
                        all columns appart from `time` remains. List of columns must be the same as was
                        for data used to fit the `predictor`
        """
        x, y = self.get_test_day_feature_vectors(test_day_id, columns)
        y_pred = self.predict_day_from_features(x, predictor)
        fig = plt.figure(figsize=(19,6))
        ax = fig.add_subplot(1, 1, 1)
        ax.title.set_text('day %d - mse=%.0f' % (test_day_id, mean_squared_error(y_pred, y)))
        l1, = plt.plot(y)
        l2, = plt.plot(y_pred)
        plt.show()
        
    def show_n_days_prediction(self, predictor, columns_to_keep=None, days=6, time_steps_back=3):
        """
        Shows plots for several testing days for given predictor.
        :param predictor: fitted sklearn predictor
        :param columns: List of columns that should remain in generated features. Deafult is None, when 
                        all columns appart from `time` remains. List of columns must be the same as was
                        for data used to fit the `predictor`
        :param days: Can be integer - than this number of random days is plotted
                            list - than days on IDs specified in list are plotted
                            string `all` than all testing days are plotted
        """
        days_list = self.get_testing_days()
        if days == 'all':
            n_days = len(days_list)
            days_list_indices = list(range(n_days-1))
        elif isinstance(days, list):
            n_days = len(days)
            days_list_indices = days
        elif isinstance(days, int):
            n_days = days
            n_data = len(days_list)
            days_list_indices = random.sample(range(0, n_data), days)
        else:
            print('show_n_days_prediction error: Not valid option days for `%s`' % (str(days)))
            return

        rows = n_days//2
        columns = 2
        if len(days_list) == 1:
            rows = 1
            columns = 1

        fig = plt.figure(figsize=(19,5*rows))
        for i, day_id in enumerate(days_list_indices):
            x, y = self.get_test_day_feature_vectors(day_id, columns_to_keep, time_steps_back)
            y_pred = self.predict_day_from_features(x, predictor, time_steps_back)
            ax = fig.add_subplot(rows, columns, i+1)
            ax.title.set_text('Day %d - mse=%.0f' % (day_id, mean_squared_error(y_pred, y)))
            l1, = plt.plot(y)
            l2, = plt.plot(y_pred)
        plt.show()

    def plot_days(self, days_list):
        """
        Plots all days attandance from `days_list`.
        :param days_list: List of days to plot
        """
        n_days = len(days_list)
        rows = n_days//2
        columns = 2
        if n_days == 1:
            rows = 1
            columns = 1

        fig = plt.figure(figsize=(19,4*rows))
        for i, day in enumerate(days_list):
            x, y = self.get_feature_vectors_from_days([day],['time'])
            ax = fig.add_subplot(rows, columns, i+1)
            ax.title.set_text('Day %d' % (i))
            l1, = plt.plot(y)
        plt.show()


