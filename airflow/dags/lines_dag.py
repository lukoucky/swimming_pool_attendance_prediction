from airflow import DAG
from scraper.line_scraper import LineScraper
from datetime import datetime, timedelta
from dateutil import tz
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
    dag_id='lines_scraper',
    default_args=default_args, 
    schedule_interval='*/30 05-21 * * *',
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

        exec_time = datetime.fromisoformat(str(execution_date))
        to_zone = tz.gettz('Europe/Prague')
        from_zone = tz.tzutc()

        exec_time = exec_time.replace(tzinfo=from_zone)
        my_timestamp = exec_time.astimezone(to_zone)

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
        cur = conn.cursor()

        for slot_id, reservations in enumerate(data):
            if len(reservations) > 0:
                query = f'INSERT INTO lines_usage (date, time_slot, reservation) VALUES (\'{date}\', {slot_id}, \'{",".join(reservations)}\');'
                logging.info(query)
                cur.execute(query)

        conn.commit()
        cur.close()
        conn.close()

    save_data(run_scraper())