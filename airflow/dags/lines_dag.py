from airflow import DAG
from scraper.line_scraper import LineScraper
from utils import convert_time_to_cet
from utils.database import LinesUsageDBHelper
from datetime import datetime, timedelta
from airflow.decorators import task
import logging
import psycopg2
import os


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2022, 8, 20),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,    
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    'execution_timeout': timedelta(seconds=300),
}

with DAG(
    dag_id='lines_scraper',
    default_args=default_args, 
    schedule_interval='10 3 * * *',
    catchup=False,
    tags=['scraper']
) as dag:

    @task(execution_timeout=timedelta(minutes=1), provide_context=True)
    def run_scraper():
        ls = LineScraper()
        res = ls.get_line_usage_for_today()
        logging.info(f'Lines data: {res}')
        return res

    @task(execution_timeout=timedelta(minutes=1))
    def save_data(data, execution_date=None):
        print(data)

        my_timestamp = convert_time_to_cet(execution_date)
        date = my_timestamp.strftime("%Y-%m-%d")

        db_port = os.getenv('POSTGRES_PORT')
        db_user = os.getenv('POSTGRES_USER')
        db_pass = os.getenv('POSTGRES_PASSWORD')
        db_name = os.getenv('POSTGRES_DB')

        conn = psycopg2.connect(
            database=db_name, 
            user=db_user, 
            password=db_pass, 
            host='postgres', 
            port=db_port)
        helper = LinesUsageDBHelper(date, data)
        helper.save_lines_usage(conn)
        conn.close()

    save_data(run_scraper())
