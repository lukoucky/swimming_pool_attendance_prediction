import os
import logging
import psycopg2


def parse_mysql_dump(dumpfile_path):
    with open(dumpfile_path) as dump_file:
        dump = dump_file.readlines()
    
    db_port = os.getenv('POSTGRES_PORT')
    db_user = os.getenv('POSTGRES_USER')
    db_pass = os.getenv('POSTGRES_PASSWORD')
    db_name = os.getenv('POSTGRES_DB')
    db_connection = psycopg2.connect(
            database=db_name, 
            user=db_user, 
            password=db_pass, 
            host='postgres', 
            port=db_port)
    cursor = db_connection.cursor()
    for line in dump:
        if line.startswith('INSERT INTO `o') or line.startswith('INSERT INTO `l'):
            line = line.replace('`', '')
            new_line = ''
            skip = False
            for character in line:
                if skip == False:
                    new_line += character
                    
                if character == '(':
                    skip = True

                if character == ',' and skip:
                    skip = False
            
            new_line = new_line.replace('lines_usage', 'lines_usage (date, time_slot, reservation)')
            new_line = new_line.replace('occupancy', 'occupancy (percent, pool, park, lines_reserved, time, day_of_week)')
            logging.info(new_line[:100])
            cursor.execute(new_line)

    db_connection.close()


def fill_database():
    dumpfile_path = os.getenv('MYSQL_DB_DUMP')
    if os.path.exists(dumpfile_path):
        parse_mysql_dump(dumpfile_path)

if __name__=='__main__':
    logging.info('#'*100)
    logging.info('fill_database')
    logging.info('#'*100)
    # fill_database()
