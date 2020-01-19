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
        # self.columns = ['pool', 'lines_reserved', 'day_of_week', 'month', 'day', 'hour', 'minute', 'holiday', 'reserved_Lavoda', 'reserved_Club Junior', 'reserved_Elab', 'reserved_Vodnik', 'reserved_Spirala', 'reserved_Amalka', 'reserved_Dukla', 'reserved_Lodicka', 'reserved_Elab team', 'reserved_Sports Team', 'reserved_Modra Hvezda', 'reserved_VSC MSMT', 'reserved_Orka', 'reserved_Activity', 'reserved_Aquamen', 'reserved_Zralok', 'reserved_SK Impuls', 'reserved_Motylek', 'reserved_3fit', 'reserved_Jitka Vachtova', 'reserved_Hodbod', 'reserved_DUFA', 'reserved_The Swim', 'reserved_Neptun', 'reserved_Strahov Cup', 'reserved_Apneaman', 'reserved_Michovsky', 'reserved_Betri', 'reserved_Pospisil', 'reserved_Vachtova', 'reserved_Riverside', 'reserved_Vodni polo Sparta', 'reserved_Road 2 Kona', 'reserved_Water Polo Sparta Praha', 'reserved_Sucha', 'reserved_Totkovicova', 'reserved_DDM Spirala', 'reserved_PS Perla', 'reserved_Dufkova - pulka drahy', 'reserved_Pavlovec', 'reserved_Sidorovich', 'reserved_OS DUFA',  'reserved_other', 'minute_of_day', 'year']
        self.prediction_steps = 288

    def get_weather(self):
        """
        Downloads actual weather forcast. It is necessary to fill valid API Key to WEATHER_URL for correnct functionality.
        """
        r = requests.get(url = WEATHER_URL) 
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

    def prepare_feature_vectors(self, columns, data_dict, time_steps_back, lines_reservations=None):
        """
        Generates feature vector for prediction algorithms
        :param columns: List with names of columns that must be in feature vector
        :param data_dict: Dictionary with time data containing columns names and default vaules
        :param time_steps_back: Number of time steps for one input to prediction algorithm
        :return: Arrays x and y with prepared features in x
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
        x,y = self.dh.build_timeseries(df, time_steps_back)
        return self.dh.reformat_feature_list([x], [y], True)

    def predict2(self, predictor, columns, data_dict, time_steps_back, lines_reservations=None):
        
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
        y_pred = self.dh.predict_day_from_features(x, predictor, time_steps_back)
        return y_pred
    
    def daterange(self, start_date, end_date):
        """
        Generator of dates from start_date to end_date
        :param start_date: Datetime.date of first date to be generated
        :param end_date: Datetime.date where to stop generator. This day will not be generated.
        """
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)
            
    def export_prediction(self, y_pred, day_of_week, month, day):
        """
        Exports predictions to CSV file
        """
        s = 'time,pool,lines_reserved\n'
        pred_id = 0
        slot_id = 0
        for hour in range(24):
            for minute in range(0,60,5):
                # prediction = 0
                # if day_of_week > 4:
                #     if slot_id > 105 and hour < 22:
                #         prediction = y_pred[pred_id]
                #         pred_id+=1
                # else:
                #     if slot_id > 57 and hour < 22:
                #         prediction = y_pred[pred_id]
                #         pred_id+=1
                prediction = 0
                if hour < 23:
                    prediction = int(y_pred[slot_id])
                s += '2020-%02d-%02d %02d:%02d:00,%d,0\n'%(month,day,hour,minute,prediction)
                slot_id += 1
        fn = '2020-%02d-%02d.csv'%(month,day)
        text_file = open(fn, "w")
        text_file.write(s)
        text_file.close()

    def generate_predictions(self, estimator, start_date, end_date, export_folder, columns=['pool','day_of_week','month','hour','minute'], time_steps_back=3):
        """
        Generates prediction vector and saves it to CSV file recognized by web server
        """
        for single_day in self.daterange(start_date, end_date):
            if single_day.weekday() > 4:
                start_minute = 0
                start_hour = 0
                self.prediction_steps = 288
            else:
                self.prediction_steps = 288
                start_hour = 0
                start_minute = 0
                
            data_dict = {
                'month' : int(single_day.strftime('%m')),
                'day' : int(single_day.strftime('%d')),
                'hour' : start_hour,
                'minute' : 0,
                'minute_of_day' : start_minute,
                'year' : int(single_day.strftime('%Y')),
                'day_of_week' : single_day.weekday(),
                'temperature_binned':3,
                'wind_binned':1,
                'humidity_binned':3,
                'precipitation_binned':0,
                'pressure_binned':2
            }
            
            for column in columns:
                if column.startswith('reserved'):
                    data_dict[column] = 0
            # x, y = self.prepare_feature_vectors(columns, data_dict, time_steps_back)
            #x, y = self.prepare_feature_vectors(columns, data_dict, time_steps_back,self.get_lines_usage_for_day(single_day))
            if start_date.weekday() < 5:
                # y_pred = self.dh.predict_day_from_features(x, estimator.weekday_model, time_steps_back)
                y_pred = self.predict2(estimator.weekday_model, columns, data_dict, time_steps_back, self.get_lines_usage_for_day(single_day))
            else:
                # y_pred = self.dh.predict_day_from_features(x, estimator.weekend_model, time_steps_back)
                y_pred = self.predict2(estimator.weekend_model, columns, data_dict, time_steps_back, self.get_lines_usage_for_day(single_day))
                print(y_pred)
            self.export_prediction(y_pred, single_day.weekday(), int(single_day.strftime('%m')), int(single_day.strftime('%d')))

if __name__ == '__main__':
    p = Predictor()
    d = datetime.now()
    clf = DoubleExtraTreesRegressor()

    # clf.fit_on_training_set()
    # clf.save_model()
    ########################################################
    # Or uncomanet line below and load already trained model
    ########################################################
    clf.load_model()
    # clf.show_n_predictions(2)
    for i in range(19):
        start_date = datetime.now() + timedelta(days=i) - timedelta(days=19)
        end_date = datetime.now()+ timedelta(days=i+1) - timedelta(days=19)
        p.generate_predictions(clf, start_date, end_date, 'export_'+clf.name, clf.columns, clf.time_steps_back)