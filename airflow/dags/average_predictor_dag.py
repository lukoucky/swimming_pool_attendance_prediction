from airflow import DAG
from predictor.monthly_average import MonthlyAveragePredictor
from datetime import datetime, timedelta
from airflow.decorators import task
import logging


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

    predictor = MonthlyAveragePredictor(datetime.now())

    @task(execution_timeout=timedelta(minutes=1))
    def get_data():
        data = predictor.get_data()
        logging.info(f'Got data from DB')
        return data

    @task(execution_timeout=timedelta(minutes=1))
    def predict(data):
        final_data = predictor.generate_prediction(data)
        logging.info(f'Goenerated prediction: {final_data}')
        return final_data

    @task(execution_timeout=timedelta(minutes=1))
    def export_csv(data):
        predictor.export_csv(data)
        logging.info('Exported data to CSV')

    export_csv(predict(get_data()))
