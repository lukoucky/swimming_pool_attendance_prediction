from datetime import datetime, timedelta
from dateutil import tz

def convert_time_to_cet(time_to_change):
    exec_time = datetime.fromisoformat(str(time_to_change))
    to_zone = tz.gettz('Europe/Prague')
    from_zone = tz.tzutc()

    exec_time = exec_time.replace(tzinfo=from_zone)
    return exec_time.astimezone(to_zone)
