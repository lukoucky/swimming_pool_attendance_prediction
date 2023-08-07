from airflow import DAG
from scraper.occupancy_scraper import OccupancyScraper
from utils.database import OccupancyDatabaseHelper
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
    'execution_timeout': timedelta(seconds=290),
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
    def save_data_to_database(data):
        db_helper = OccupancyDatabaseHelper()
        db_helper.save_occupancy(data)
        return data
    
    @task(execution_timeout=timedelta(minutes=1))
    def export_csv():
        export_time = datetime.strptime(datetime.now().strftime('%Y-%m-%d')+' 00:00:00', '%Y-%m-%d %H:%M:%S')
        db_helper = OccupancyDatabaseHelper()
        dt = datetime.now().strftime('%Y-%m-%d')
        file_path = f'/web_data/{dt}.csv'
        csv_string = 'time,pool,lines_reserved\n'

        lines_usage = db_helper.get_lines_usage_vector_for_day(dt)
        occupancy = db_helper.get_occupancy_vector_for_day(export_time)

        for pool, lines in zip(occupancy, lines_usage):
            this_time_string = export_time.strftime('%Y-%m-%d %H:%M:%S')
            export_time = export_time + timedelta(minutes=5)
            csv_string += f'{this_time_string},{pool},{lines}\n'
        
        with open(file_path, 'w') as csv_file:
            csv_file.write(csv_string)

    save_data_to_database(run_scraper()) >>  export_csv()
