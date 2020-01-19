from models.tree_models import MyExtraTreesRegressor, DoubleExtraTreesRegressor
from models.monthly_average import MonthlyAverageClassifier
from data_helper import DataHelper
from days_statistics import DaysStatistics
from datetime import timedelta, date, datetime
import pandas as pd
import os
import json
import requests
from bs4 import BeautifulSoup
import urllib.request
import sys
import unicodedata
from utils import Day

WEATHER_API_KEY = None

# URL is missing API KYE. Full address with API KEY is constructed in get_weather() method
WEATHER_URL = 'http://api.openweathermap.org/data/2.5/forecast?id=3067696&units=metric&appid='


class Predictor():
    
    def __init__(self):
        """
        Constructor
        """
        self.dh = DataHelper()
        self.ds = DaysStatistics()
        self.prediction_steps = 288

    def get_weather(self):
        """
        Downloads actual weather forcast. It is necessary to fill valid API Key to WEATHER_URL for correnct functionality.
        """
        r = requests.get(url = WEATHER_URL+WEATHER_API_KEY) 
        json_data = json.loads(r)
        weather = []
        for weather_prediction in json_data['list']:
            time = datetime.fromtimestamp(weather_prediction['dt'])
            temperature = weather_prediction['main']['temp']
            wind = weather_prediction['wind']['speed']

            rain, snow = 0.0, 0.0
            if 'rain' in weather_prediction['main']:
                 rain = weather_prediction['main']['rain']['3h']
            if 'snow' in weather_prediction['main']:
                 rain = weather_prediction['main']['snow']['3h']

            precipitation = rain + snow
            pressure =  weather_prediction['main']['pressure']
            weather.append([time, temperature,wind,precipitation,pressure])
        return weather
        
    def get_date_string(self, date):
        """
        Generates part of URL address for Sutka pool web page with table of reserved lines for given date.
        :param date: Datetime.date
        :return: The rest of URL that starts with: http://www.sutka.eu/obsazenost-bazenu
        """
        next_day = date + timedelta(days=1)
        day_str = date.strftime('%d.%m.%Y')
        next_day_str = next_day.strftime('%d.%m.%Y')
        addr_str = '?from='+day_str+'&to='+next_day_str+'&send=Zobrazit&do=form-submit'
        return addr_str

    def parse_line_usage_table(self, table_rows):
        """
        Parse table with reserved lines from web page http://www.sutka.eu/obsazenost-bazenu
        :param table_rows: HTML table from web that is prepared in method get_lines_usage_for_day
        :return: list with 64 items. Each item is 15 minute time slot starting from 6:00
        """
        time_slots = ['']*64
        i = 0
        for i in range(1,9): 
            row = table_rows[i]
            slot = 0
            row_str = '|'
            th = row.find_all('th')

            cols = row.find_all('td')
            for col in cols:
                if len(col) == 3:
                    title = unicodedata.normalize('NFKD', col.attrs['title']).encode('ascii','ignore')
                    if 'colspan' in col.attrs:
                        row_str += ' * | * |'
                        time_slots[slot] += title.decode('utf-8')+','
                        time_slots[slot+1] += title.decode('utf-8')+','
                        slot += 2
                    else:
                        row_str += ' * |'
                        time_slots[slot] += title.decode('utf-8')+','
                        slot += 1
                else:
                    row_str += '   |'
                    slot += 1
        time_slots[63] = time_slots[61]
        time_slots[62] = time_slots[61]
        time_slots[61] = time_slots[60]
        return time_slots

    def get_lines_usage_for_day(self, day):
        """
        Generates list with lines reservations for the day
        :param day: Datetime.date of interest
        :return: list with 64 items. Each item is 15 minute time slot starting from 6:00
        """
        date_str = self.get_date_string(day)
        r = urllib.request.urlopen('http://www.sutka.eu/obsazenost-bazenu'+date_str).read()
        soup = BeautifulSoup(r, 'html.parser')
        table = soup.find('table', attrs={'class':'pooltable'})
        table_rows = table.find_all('tr')
        return self.parse_line_usage_table(table_rows)

    def predict(self, predictor, columns, data_dict, time_steps_back, lines_reservations=None):
        """
        Prepares feature vector for prediction algorithms an generates prediction
        :param predictor: sklearn predictor or other with simmilar interface
        :param columns: List with names of columns that must be in feature vector
        :param data_dict: Dictionary with time data containing columns names and default vaules
        :param time_steps_back: Number of time steps for one input to prediction algorithm
        :param lines_reservations: List with line reservations for predicted day generated by get_lines_usage_for_day method
        :return: Vector with prediction for the day (288 items)
        """
        lines_reserved_id = -1
        org_ids = dict()
        for org_id, column in enumerate(columns):
            if column.startswith('reserved_'):
                org_ids[column] = org_id
            if column == 'lines_reserved':
                lines_reserved_id = org_id

        if lines_reservations is None:
            lines_reservations = ['']*64
            
        data = list()
        for i in range(self.prediction_steps):
            data.append([0]*len(columns))
            for j, column in enumerate(columns):
                if column in data_dict:
                    data[i][j] = data_dict[column]

            slot_id = (data_dict['minute_of_day']-360)//15
            if slot_id >= 0 and slot_id < 64:
                org_list = lines_reservations[slot_id].split(',')[:-1]
                for name in org_list:
                    feature_name = 'reserved_' + name
                    if feature_name in columns:
                        data[i][org_ids[feature_name]] += 1
                    elif 'reserved_other' in columns:
                        data[i][org_ids['reserved_other']] += 1
                        
                    if lines_reserved_id >= 0:
                        data[i][lines_reserved_id] += 1
                        
            data_dict['minute'] += 5
            data_dict['minute_of_day'] += 5
            if data_dict['minute'] == 60:
                data_dict['minute'] = 0
                data_dict['hour'] += 1

        df = pd.DataFrame(data, columns=columns) 
        day = Day('ts')
        day.data = df
        x, y = self.dh.get_feature_vectors_from_days([day], [], time_steps_back, 1, True)
        return self.dh.predict_day_from_features(x, predictor, time_steps_back)
    
    def daterange(self, start_date, end_date):
        """
        Generator of dates from start_date to end_date
        :param start_date: Datetime.date of first date to be generated
        :param end_date: Datetime.date where to stop generator. This day will not be generated.
        """
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)
            
    def export_prediction(self, y_pred, prediction_date, export_folder):
        """
        Exports predictions to CSV file
        :param y_pred: Vecotr of 288 items with prediction
        :param prediction_date: Datetime with prediction date
        :param export_folder: Directory where to store csv with prediction. Will be created if does not exist
        """
        if not os.path.isdir(export_folder):
            os.mkdir(export_folder)

        day_of_week = prediction_date.weekday()
        month = int(prediction_date.strftime('%m'))
        day = int(prediction_date.strftime('%d'))
        s = 'time,pool,lines_reserved\n'
        pred_id = 0
        slot_id = 0
        for hour in range(24):
            for minute in range(0,60,5):
                prediction = 0
                if hour < 23:
                    prediction = int(y_pred[slot_id])
                s += '2020-%02d-%02d %02d:%02d:00,%d,0\n'%(month,day,hour,minute,prediction)
                slot_id += 1
        fn = '2020-%02d-%02d.csv'%(month,day)
        text_file = open(os.path.join(export_folder, fn), "w")
        text_file.write(s)
        text_file.close()

    def generate_predictions(self, estimator, prediction_date, export_folder):
        """
        Generates prediction vector and saves it to CSV file recognized by web server
        :param estimator: Estimator class with prediction model, time_steps_back and columns members
        :param prediction_date: Datetime with prediction date
        :param export_folder: Directory where to store csv with prediction. Will be created if does not exist
        """
        data_dict = {
            'month' : int(prediction_date.strftime('%m')),
            'day' : int(prediction_date.strftime('%d')),
            'hour' : 0,
            'minute' : 0,
            'minute_of_day' : 0,
            'year' : int(prediction_date.strftime('%Y')),
            'day_of_week' : prediction_date.weekday(),
            'temperature_binned':3,
            'wind_binned':1,
            'humidity_binned':3,
            'precipitation_binned':0,
            'pressure_binned':2
        }
        
        for column in estimator.columns:
            if column.startswith('reserved'):
                data_dict[column] = 0

        if prediction_date.weekday() < 5:
            y_pred = self.predict(estimator.weekday_model, estimator.columns, data_dict, estimator.time_steps_back, self.get_lines_usage_for_day(prediction_date))
        else:
            y_pred = self.predict(estimator.weekend_model, estimator.columns, data_dict, estimator.time_steps_back, self.get_lines_usage_for_day(prediction_date))
        self.export_prediction(y_pred, prediction_date, export_folder)

if __name__ == '__main__':
    p = Predictor()
    d = datetime.now()
    clf = DoubleExtraTreesRegressor()

    if os.path.isfile('data/'+clf.name+'.pickle'):
        clf.load_model()
    else:
        clf.fit_on_training_set()
        clf.save_model()

    # Generates prediction for 5 days (today and 4 into the future)
    for i in range(5):
        prediction_date = datetime.now() + timedelta(days=i)
        p.generate_predictions(clf, prediction_date, 'export_'+clf.name)
