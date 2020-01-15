from models.tree_models import MyExtraTreesRegressor, MyExtraTreesClassifier, MyRandomForestRegressor, MyRandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from models.monthly_average import MonthlyAverageClassifier
from data_helper import DataHelper
from days_statistics import DaysStatistics
from datetime import timedelta, date, datetime
import pandas as pd
import os

class Predictor:
    """
    Class for automated predictions runs be cron on webserver
    """
    def __init__(self):
        """
        Constructor
        """
        self.dh = DataHelper()
        self.ds = DaysStatistics()
        self.prediction_steps = 210
        
    def add_estimator(self, estimator):
        """
        Adds new estimator to for predictions
        """
        self.estimators.append(estimator)
    
    def generate_predictions(self, estimator, start_date, end_date, export_folder, columns=['pool','day_of_week','month','hour','minute'], time_steps_back=3):
        """
        Generates prediction vector and saves it to CSV file recognized by web server
        """
        for single_day in self.daterange(start_date, end_date):
            data_dict = {
                'month' : int(single_day.strftime('%m')),
                'day' : int(single_day.strftime('%d')),
                'hour' : 5,
                'minute' : 0,
                'day_of_week' : single_day.weekday(),
            }
            x, y = self.prepare_feature_vectors(columns, data_dict, time_steps_back)
            y_pred = self.dh.predict_day_from_features(x, estimator, time_steps_back)
            self.export_prediction(y_pred, export_folder, single_day, time_steps_back)
            
    def export_prediction(self, y_pred, export_folder, single_day, time_steps_back=3):
        """
        Exports predictions to CSV file
        """
        df = pd.DataFrame(columns=['time','pool'])
        time_step = datetime.strptime(single_day.strftime('%Y%m%d'), '%Y%m%d')

        hour = 0
        minute = 0
        for i in range(288):
            time_step = time_step.replace(hour=hour, minute=minute)
            df.loc[i] = [time_step.strftime('%Y-%m-%d %H:%M:%S'), 0]
            minute += 5
            if minute == 60:
                minute = 0
                hour += 1

        hour = 5
        minute = 5*time_steps_back
        y_id = 0
        start_id = df[df['time'] == time_step.strftime('%Y-%m-%d')+' 05:00:00'].index.item() + time_steps_back + 1
        for i in range(start_id, len(y_pred)+start_id):
            time_step = time_step.replace(hour=hour, minute=minute)
            df.loc[i] = [time_step.strftime('%Y-%m-%d %H:%M:%S'), y_pred[y_id]]
            minute += 5
            y_id += 1
            if minute == 60:
                minute = 0
                hour += 1

        csv_file = os.path.join(export_folder, time_step.strftime('%Y-%m-%d')+'.csv')
        export_csv = df.to_csv (csv_file, index = None, header=True)

    def prepare_feature_vectors(self, columns, data_dict, time_steps_back):
        """
        Generates feature vector for prediction algorithms
        :param columns: List with names of columns that must be in feature vector
        :param data_dict: Dictionary with time data containing columns names and default vaules
        :param time_steps_back: Number of time steps for one input to prediction algorithm
        :return: Arrays x and y with prepared features in x
        """
        data = list()
        for i in range(self.prediction_steps):
            data.append([0]*len(columns))
            for j, column in enumerate(columns):
                if column in data_dict:
                    data[i][j] = data_dict[column]
            data_dict['minute'] += 5
            if data_dict['minute'] == 60:
                data_dict['minute'] = 0
                data_dict['hour'] += 1

        df = pd.DataFrame(data, columns=columns) 
        return self.dh.build_timeseries(df, time_steps_back)
    
    def daterange(self, start_date, end_date):
        """
        Generator of dates from start_date to end_date
        :param start_date: Datetime.date of first date to be generated
        :param end_date: Datetime.date where to stop generator. This day will not be generated.
        """
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def export_prediction_lines(self, export_folder, single_day):
        df = pd.DataFrame(columns=['time','pool','lines_reserved'])
        time_step = datetime.strptime(single_day.strftime('%Y%m%d'), '%Y%m%d')

        hour = 0
        minute = 0
        for i in range(288):
            time_step = time_step.replace(hour=hour, minute=minute)
            df.loc[i] = [time_step.strftime('%Y-%m-%d %H:%M:%S'), 'null', 0]
            minute += 5
            if minute == 60:
                minute = 0
                hour += 1

        csv_file = os.path.join(export_folder, time_step.strftime('%Y-%m-%d')+'.csv')
        export_csv = df.to_csv (csv_file, index = None, header=True)

p = Predictor()
columns_pred = ['pool','day_of_week','month','hour','minute']

# mac = MonthlyAverageClassifier()
# mac.fit(p.dh.get_training_days(),columns_pred)

# clf = ExtraTreesClassifier(random_state=17, n_estimators=20, max_depth=50, min_samples_split=5, min_samples_leaf=1)
# x_train, y_train, x_test, y_test = p.dh.generate_feature_vectors(columns_pred)
# clf.fit(x_train, y_train)

start_date = date(2019, 12, 1)
end_date = date(2020, 2, 1)

etr = MyExtraTreesRegressor()
etr.columns = ['pool','day_of_week','month','hour','minute']
etr.model = ExtraTreesRegressor(random_state=17, n_estimators=65, max_depth=35, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None)
etr.time_steps_back = 3
etr.fit_on_training_set()

p.generate_predictions(etr.model, start_date, end_date, 'export')
