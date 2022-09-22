from cgitb import reset
from airflow import DAG
from predictor.average import DailyAveragePredictor
from utils.database import PredictorDatabaseHelper
from datetime import datetime, timedelta
from dateutil import tz
from airflow.decorators import task
import logging
import copy


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2022, 8, 20),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,    
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    'execution_timeout': timedelta(seconds=300),
}

with DAG(
    dag_id='daily_average_predictor',
    default_args=default_args, 
    schedule_interval=None,
    catchup=False,
    tags=['predictor']
) as dag:

    @task(execution_timeout=timedelta(minutes=1))
    def get_data():
        db_helper = PredictorDatabaseHelper()
        logging.info('In average predictor')
        date_from = datetime.now() - timedelta(days = 33)
        data = db_helper.get_daily_occupancy_vectors_from(date_from, datetime.today().weekday())
        return data

    @task(execution_timeout=timedelta(minutes=1))
    def predict(data):
        all_data = [0]*288
        for vector in data.values():
            for value_id, value in enumerate(vector):
                if value is not None:
                    # TODO: Think about how to handle missing data here
                    all_data[value_id] += value

        final_data = []
        for value in all_data:
            final_data.append(value/len(data))

        return final_data

    @task(execution_timeout=timedelta(minutes=1))
    def export_csv(data):
        date = datetime.now()
        export_time = datetime.strptime(date.strftime('%Y-%m-%d')+' 00:00:00', '%Y-%m-%d %H:%M:%S')
        dt = date.strftime('%Y-%m-%d')
        file_path = f'/web_data/prediction_monthly_average/{dt}.csv'
        csv_string = 'time,pool\n'

        for pool in data:
            this_time_string = export_time.strftime('%Y-%m-%d %H:%M:%S')
            export_time = export_time + timedelta(minutes=5)
            csv_string += f'{this_time_string},{pool}\n'
        
        with open(file_path, 'w') as csv_file:
            csv_file.write(csv_string)

    export_csv(predict(get_data()))
