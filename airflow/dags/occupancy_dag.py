from airflow import DAG
from occupancy_scraper import OccupancyScraper
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
    schedule_interval='*/1 * * * *',
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

        print(f'{pool} {park} {occupancy}')
        # db = MySQLdb.connect("127.0.0.1",user,passwd,db_name)
        # cursor = db.cursor()

        # table_name = 'occupancy'

        # dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # dow = datetime.datetime.today().weekday()

        # print("Current occupancy: %d%%, In pool: %d, In aquapark: %d (Time: %s)"%(occupancy, pool, park, dt))

        # t = "INSERT INTO %s (percent, pool, park, lines_reserved, time, day_of_week) VALUES (%s,%s,%s,%d,\'%s\',%d);"%(table_name, occupancy, pool, park, lines, dt, dow)
        # cursor.execute(t)

        # db.commit()
        # db.close()

    save_data(run_scraper())
