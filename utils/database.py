import os
import logging
import psycopg2
from typing import Optional
from datetime import datetime, timedelta
from utils import convert_time_to_cet


class DatabaseHelper:
    def __init__(self) -> None:
        db_port = os.getenv('POSTGRES_PORT')
        db_user = os.getenv('POSTGRES_USER')
        db_pass = os.getenv('POSTGRES_PASSWORD')
        db_name = os.getenv('POSTGRES_DB')
        self.db_connection = psycopg2.connect(
                database=db_name, 
                user=db_user, 
                password=db_pass, 
                host='postgres', 
                port=db_port)
    
    def get_cursor(self):
        return self.db_connection.cursor()

    def execute(self, query):
        cursor = self.get_cursor()
        logging.info(query)
        cursor.execute(query)

        if query.lower().startswith('insert') or query.lower().startswith('update'):
            self.db_connection.commit()
            result = None
        else:
            result = cursor.fetchall()
        cursor.close()
        return result

    def __del__(self) -> None:
        self.db_connection.close()


class LinesUsageDBHelper:
    def __init__(self, date, reservations) -> None:
        self.date = date
        self.reservations = reservations

    def save_to_database(self, date, slot_id, reservation, cursor):
        query = f'INSERT INTO lines_usage (date, time_slot, reservation) VALUES (\'{date}\', {slot_id}, \'{reservation}\');'
        logging.info(query)
        cursor.execute(query)

    def update_entry(self, row_id, reservation, cursor):
        query = f'UPDATE lines_usage SET reservation = \'{reservation}\' WHERE id = {row_id};'
        logging.info(query)
        cursor.execute(query)

    def get_id_for_line_usage(self, date, slot_id, cursor):
        query = f'SELECT id FROM lines_usage WHERE date = \'{date}\' AND time_slot = {slot_id};'
        cursor.execute(query)
        result = cursor.fetchone()

        if result is None:
            logging.info(f'No data for {query}')
            return None
        else:
            logging.info(f'Id {result[0]} for {query}')
            return result[0]

    def save_lines_usage(self, db_connection):
        cursor = db_connection.cursor()
        for slot_id, reservation in enumerate(self.reservations):
            if len(reservation) > 0:
                reservation_string = ','.join(reservation)
                row_id = self.get_id_for_line_usage(self.date, slot_id, cursor)
                if row_id is None:
                    self.save_to_database(self.date, slot_id, reservation_string, cursor)
                else:
                    self.update_entry(row_id, reservation_string, cursor)
        db_connection.commit()
        cursor.close()


class OccupancyDatabaseHelper:
    def __init__(self) -> None:
        self.db_helper = DatabaseHelper()

    def save_occupancy(self, data):
        occupancy = int(data["percentage"])
        pool  = int(data["pool"])
        park =  int(data["park"])
        my_timestamp = convert_time_to_cet(datetime.now())

        query = f'INSERT INTO occupancy (percent, pool, park, time, day_of_week) VALUES ({occupancy}, {pool}, {park}, \'{my_timestamp.strftime("%Y-%m-%d %H:%M:%S")}\', {datetime.today().weekday()});'
        self.db_helper.execute(query)

    def _get_data(self, date: datetime, value_index: int, time_index: int=5) -> dict:
        next_date = date + timedelta(days=1)
        query = f'SELECT * FROM occupancy WHERE time >= \'{date}\' and time < \'{next_date}\' ORDER BY time ASC;'
        result = self.db_helper.execute(query)

        day_data = self.parse_day_data_to_vectors(result, time_index, value_index)
        if date.date() in day_data:
            return day_data[date.date()]
        else:
            return ['nan']*288

    def get_occupancy_vector_for_day(self, date: datetime):
        return self._get_data(date, 2)
    
    def get_lines_usage_vector_for_day(self, date: datetime):
        return self._get_data(date, 4)

    @staticmethod
    def get_time_slot_for_datetime(timestamp: datetime) -> int:
        """
        From datatime object export time slot for the 5 minute split day
        :param timestamp: Datetime time stamp
        :return: Slot id for day split after 5 minutes to array
        """
        minute_of_day = timestamp.hour*60 + timestamp.minute
        return round(minute_of_day/5)

    def parse_day_data_to_vectors(self, day_data: list, time_index: int, value_index: int) -> dict:
        """
        From the output of database query to vector of 288 values
        :param day_data: Query results
        :param time_index: 
        :param value_index:
        :return: Dictionary where key is the datatime.date and value is
                 288 items long array.
        """
        days = {}
        for row in day_data:
            date = row[time_index].date()
            time_slot = OccupancyDatabaseHelper.get_time_slot_for_datetime(row[time_index])

            if date not in days:
                days[date] = ['nan']*288
        
            days[date][time_slot] = row[value_index]
        
        return days


class PredictorDatabaseHelper:
    def __init__(self) -> None:
        self.db_helper = DatabaseHelper()
    
    def get_daily_data_from(self, date_from: datetime, day_of_week: Optional[int] = None) -> dict:
        """
        Returns daily data from `date_from` up to the newest data in DB.
        :param date_from: Date from when to get data
        :param day_of_week: Optional day of week filter. If None all days 
                            from `date_from` are returned. If number from 
                            0 to 6 returns just the selected days of week.
        :return: Dictionary where keys are dates and values are 288 elements
                 long lists with data in this format:
                 {
                    datetime(2022, 08, 01): {
                        'occupancy': [0,0,0,.....],
                        'lines': ['Team1', 'Team1, Team2', ...]
                    }
                 }
        """
        return dict()

    def get_daily_occupancy_vectors_from(self, date_from: datetime, day_of_week: Optional[int] = None) -> list:
        """
        Returns daily data from `date_from` up to the newest data in DB as
        a list of vectors with 288 elements (one for each 5 minutes of a day)
        :param date_from: Date from when to get data
        :param day_of_week: Optional day of week filter. If None all days 
                            from `date_from` are returned. If number from 
                            0 to 6 returns just the selected days of week.
        :return: List of vectors each with 288 elements (one for each 5 minutes of a day)
        """
        delta = datetime.now() - date_from
        if day_of_week is None:
            query = f'SELECT * FROM occupancy WHERE time > current_date - interval \'{delta.days}\' day ORDER BY time ASC;'
        else:
            query = f'SELECT * FROM occupancy WHERE time > current_date - interval \'{delta.days}\' day AND day_of_week = {day_of_week} ORDER BY time ASC;'
        query_result = self.db_helper.execute(query)
        
        return self.parse_days_occupancy_to_vectors(query_result)

    @staticmethod
    def get_time_slot_for_datetime(timestamp: datetime) -> int:
        """
        From datatime object export time slot for the 5 minute split day
        :param timestamp: Datetime time stamp
        :return: Slot id for day split after 5 minutes to array
        """
        minute_of_day = timestamp.hour*60 + timestamp.minute
        return round(minute_of_day/5)

    def parse_days_occupancy_to_vectors(self, day_data: list) -> dict:
        """
        From the output of database query to occupancy table 
        return each day occupancy as a vector of 288 values.
        :param day_data: Query results
        :return: Dictionary where key is the datatime.date and value is
                 288 items long array.
        """
        days = {}
        for row in day_data:
            date = row[5].date()
            time_slot = PredictorDatabaseHelper.get_time_slot_for_datetime(row[5])

            if date not in days:
                days[date] = [None]*288
        
            days[date][time_slot] = row[2]
        
        return days
