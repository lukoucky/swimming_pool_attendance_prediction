from sklearn.ensemble import ExtraTreesClassifier
from models.monthly_average import MonthlyAverageClassifier
from data_helper import DataHelper
from days_statistics import DaysStatistics
from datetime import timedelta, date


class Predictor:
    def __init__(self):
        self.dh = DataHelper()
        self.ds = DaysStatistics()
        self.estimators = list()
        
    def add_estimator(self, estimator):
        self.estimators.append(estimator)
    
    def generate_predictions(self, start_date, end_date, columns=None, time_steps_back=3):
        for single_day in self.daterange(start_date, end_date):
            month = int(single_day.strftime("%Y-%m-%d"))
            data = [['tom', 10], ['nick', 15], ['juli', 14]] 
            print(single_day.strftime("%Y-%m-%d"), single_day.weekday())
            
            dh.build_timeseries(self, data_frame, time_steps_back)
            
    
    def daterange(self, start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

start_date = date(2019, 11, 1)
end_date = date(2019, 11, 10)
p = Predictor()
p.generate_predictions(start_date, end_date)
