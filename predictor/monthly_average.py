from utils.database import PredictorDatabaseHelper
from datetime import datetime, timedelta


class MonthlyAveragePredictor:
    def __init__(self, date: datetime) -> None:
        self.date = date

    def get_data(self):
        db_helper = PredictorDatabaseHelper()
        date_from = self.date - timedelta(days = 33)
        data = db_helper.get_daily_occupancy_vectors_from(date_from, self.date.weekday())
        return data

    def generate_prediction(self, data):
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
    
    def export_csv(self, data):
        export_time = datetime.strptime(self.date.strftime('%Y-%m-%d')+' 00:00:00', '%Y-%m-%d %H:%M:%S')
        dt = self.date.strftime('%Y-%m-%d')
        file_path = f'/web_data/prediction_monthly_average/{dt}.csv'
        csv_string = 'time,pool\n'

        for pool in data:
            this_time_string = export_time.strftime('%Y-%m-%d %H:%M:%S')
            export_time = export_time + timedelta(minutes=5)
            csv_string += f'{this_time_string},{pool}\n'
        
        with open(file_path, 'w') as csv_file:
            csv_file.write(csv_string)
