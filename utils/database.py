import logging


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
