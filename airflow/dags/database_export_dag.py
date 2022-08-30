from calendar import month
from airflow import DAG
from datetime import date, datetime, timedelta
from dateutil import tz
from airflow.decorators import task
from subprocess import PIPE, Popen
import dropbox
import logging
import shlex
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

dump_file = 'database_dump.sql'

with DAG(
    dag_id='database_backup',
    default_args=default_args, 
    schedule_interval='0 4 * * *',
    catchup=False,
    tags=['database']
) as dag:

    @task(execution_timeout=timedelta(minutes=10))
    def export_database():
        """Export database backup from postgres"""
        db_port = os.getenv('POSTGRES_PORT')
        db_user = os.getenv('POSTGRES_USER')
        db_name = os.getenv('POSTGRES_DB')
        dump_script = f'pg_dump -d {db_name} -h postgres -t lines_usage -t occupancy -p {db_port} -U {db_user} -f {dump_file}'
        logging.info(f'Dumping datbase to {dump_file}')
        command = shlex.split(dump_script)

        p = Popen(command, shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        logging.info(stdout)
        if len(stderr) > 0:
            logging.error(stderr)
        
        if not os.path.exists(dump_file):
            logging.error(f'File {dump_file} does not exist')
            assert False, f'File {dump_file} does not exist'

    @task(execution_timeout=timedelta(minutes=10))
    def backup_to_cloud():
        """Upload exported backup to dropbox"""
        exec_time = datetime.now()
        to_zone = tz.gettz('Europe/Prague')
        from_zone = tz.tzutc()

        exec_time = exec_time.replace(tzinfo=from_zone)
        my_timestamp = exec_time.astimezone(to_zone)

        # dropbox_access_token = os.getenv('DROPBOX_ACCESS_TOKEN')
        dropbox_refresh_token = os.getenv('DROPBOX_REFRESH_TOKEN')
        dropbox_path = f'/db_backup_{my_timestamp.strftime("%Y-%m-%dT%H:%M:%S")}.sql'
        dropbox_app_key = os.getenv('DROPBOX_APP_KEY')
        dropbox_app_secret = os.getenv('DROPBOX_APP_SECRET')
        logging.info(f'Uploading {dump_file} to Dropbox as {dropbox_path}')
        client = dropbox.Dropbox(
                            app_key = dropbox_app_key,
                            app_secret = dropbox_app_secret,
                            oauth2_refresh_token = dropbox_refresh_token
                        )
        client.files_upload(open(dump_file, "rb").read(), dropbox_path)
    
    @task(execution_timeout=timedelta(minutes=1))
    def remove_old_backups():
        """Remove backup older then one month to keep only few backup files"""
        last_date_to_keep = datetime.now()-timedelta(days=31)
        last_date_to_keep = last_date_to_keep.strftime("%Y-%m-%d")
        logging.info(f'Removing database backups older then {last_date_to_keep}')
        # TODO: List files from dropbox and remove old ones

    export_database() >> backup_to_cloud() >> remove_old_backups()
