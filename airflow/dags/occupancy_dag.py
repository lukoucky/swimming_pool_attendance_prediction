from airflow import DAG
from scraper.occupancy_scraper import OccupancyScraper
from utils import convert_time_to_cet
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
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    'execution_timeout': timedelta(seconds=300),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

with DAG(
    dag_id='occupancy_scraper',
    default_args=default_args, 
    schedule_interval='*/5 * * * *',
    catchup=False,
    tags=['scraper']
) as dag:

    @task(execution_timeout=timedelta(minutes=1))
    def run_scraper():
        os = OccupancyScraper()
        res = os.get_current_occupancy()
        logging.info(f'Occupancy data: pool: {res["pool"]}, park: {res["park"]}, lines: {res["lines"]}, percent: {res["percentage"]}')
        return res

    @task(execution_timeout=timedelta(minutes=1))
    def save_data(data):
        occupancy = int(data["percentage"])
        pool  = int(data["pool"])
        park =  int(data["park"])

        my_timestamp = convert_time_to_cet(datetime.now())

        db_port = os.getenv('POSTGRES_PORT')
        db_user = os.getenv('POSTGRES_USER')
        db_pass = os.getenv('POSTGRES_PASSWORD')
        db_name = os.getenv('POSTGRES_DB')

        query = f'INSERT INTO occupancy (percent, pool, park, time, day_of_week) VALUES ({occupancy}, {pool}, {park}, \'{my_timestamp.strftime("%Y-%m-%d %H:%M:%S")}\', {datetime.today().weekday()});'
        logging.info(query)
        conn = psycopg2.connect(
            database=db_name, 
            user=db_user, 
            password=db_pass, 
            host='postgres', 
            port=db_port)
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()

        cur.close()
        conn.close()

    save_data(run_scraper())
