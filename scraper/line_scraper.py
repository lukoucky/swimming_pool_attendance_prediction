from bs4 import BeautifulSoup
import datetime
import urllib.request
import unicodedata


class LineScraper:
    def __init__(self) -> None:
        self.line_usage_url = 'http://www.sutka.eu/obsazenost-bazenu'

    def _scrape_line_usage_old(self, table_rows):
        """
        Scrapes line usage for dates BEFORE 1.12.2018
        :return: List with 64 items - one for each 15 segment when pool is open. 
                 Each item contains list of groups that reserved lines in given 15 minute slot.
                 One name can be represented multiple times in one time slot. This means that the 
                 group have more lines reserved. One name instance means one reserved line.
        """
        time_slots = ['']*64
        i = 0
        for i in range(1,9): 
            row = table_rows[i]
            slot = 0
            row_str = '|'
            cols = row.find_all('td')
            count_this = True
            for col in cols:
                if count_this:
                    count_this = False
                    if len(col) == 3:
                        title = unicodedata.normalize('NFKD', col.attrs['title']).encode('ascii','ignore').decode('utf-8')
                        row_str += ' * |'
                        time_slots[slot] += title+','
                        time_slots[slot+1] += title+','
                        slot += 2
                    else:
                        row_str += '   |'
                        slot += 2
                else:
                    count_this = True

        # Last 2 slots are per 30 minute, the rest are per 15 minutes
        # this hack will make all slots 15 minutes
        time_slots[63] = time_slots[61]
        time_slots[62] = time_slots[61]
        time_slots[61] = time_slots[60]
        return time_slots

    def _scrape_line_usage_new(self, table_rows):
        """
        Scrapes line usage for dates AFTER 1.12.2018
        :return: List with 64 items - one for each 15 segment when pool is open. 
                 Each item contains list of groups that reserved lines in given 15 minute slot.
                 One name can be represented multiple times in one time slot. This means that the 
                 group have more lines reserved. One name instance means one reserved line.
        """
        time_slots = [ [] for _ in range(64) ]
        print(time_slots)
        # time_slots = ['']*64
        i = 0
        for i in range(1,9): 
            row = table_rows[i]
            slot = 0
            row_str = '|'

            cols = row.find_all('td')
            for col in cols:
                if len(col) == 3:
                    title = unicodedata.normalize('NFKD', col.attrs['title']).encode('ascii','ignore').decode('utf-8')
                    if 'colspan' in col.attrs:
                        row_str += ' * | * |'
                        time_slots[slot].append(title)
                        time_slots[slot+1].append(title)
                        slot += 2
                    else:
                        row_str += ' * |'
                        time_slots[slot].append(title)
                        slot += 1
                else:
                    row_str += '   |'
                    slot += 1

        # Last 2 slots are per 30 minute, the rest are per 15 minutes
        # this hack will make all slots 15 minutes
        time_slots[63] = time_slots[61]
        time_slots[62] = time_slots[61]
        time_slots[61] = time_slots[60]
        return time_slots

    @staticmethod
    def _get_date_url_string(date: datetime.datetime):
        """
        Generates query string for the lines reservation page for selected date
        :param date: Requested date of lines reservation page
        :return: Query string of lines reservation page for selected date
        """
        next_day = date+datetime.timedelta(days=1)
        day_str = date.strftime('%d.%m.%Y')
        next_day_str = next_day.strftime('%d.%m.%Y')
        addr_str = '?from='+day_str+'&to='+next_day_str+'&send=Zobrazit&do=form-submit'
        return addr_str

    def get_line_usage_for_day(self, day: datetime.datetime):
        """
        Scrapes line usage for selected day.
        :param day: Datetime object of selected day
        :return: List with 64 lists - one for each 15 minute reservation slot.
                 Each list contains names of groups that reserved lines in given 15 minute slot.
                 One name can be represented multiple times in one time slot. This means that the 
                 group have more lines reserved. One name instance means one reserved line.
        """
        date_str = LineScraper._get_date_url_string(day)
        r = urllib.request.urlopen(self.line_usage_url+date_str).read()
        soup = BeautifulSoup(r, 'html.parser')
        table = soup.find('table', attrs={'class':'pooltable'})
        table_rows = table.find_all('tr')

        change_day = datetime.datetime(2018,12,1)
        if(day < change_day):
            return self._scrape_line_usage_old(table_rows)
        else:
            return self._scrape_line_usage_new(table_rows)
    
    def get_line_usage_for_today(self):
        """
        Scrapes line usage for today.
        :return: List with 64 lists - one for each 15 minute reservation slot.
                 Each list contains names of groups that reserved lines in given 15 minute slot.
                 One name can be represented multiple times in one time slot. This means that the 
                 group have more lines reserved. One name instance means one reserved line.
        """
        day = datetime.datetime.now()
        return self.get_line_usage_for_day(day)

if __name__=='__main__':
    ls = LineScraper()
    print(ls.get_line_usage_for_today())
