from bs4 import BeautifulSoup
import datetime
import urllib.request


class OccupancyScraper:
    def __init__(self) -> None:
        self.occupancy_url = 'http://www.sutka.eu/'
        self.line_usage_url = 'http://www.sutka.eu/obsazenost-bazenu'

    def _scrape_current_occupancy_data(self):
        """
        Scrapes current pool and park occupancy 
        :return: List with four items: Info text, occupancy percentage, people in pool, people in park
                 In case of some problems with parsing None is returned.
        """
        r = urllib.request.urlopen(self.occupancy_url).read()
        soup = BeautifulSoup(r, 'html.parser')
        header_info = soup.find('div', attrs={'class':'header-info'})
        data = header_info.find_all('strong')
        # There are 4 strong items: Info text, occupancy percentage, people in pool, people in park

        if len(data) == 4:
            return data
        else:
            return None

    def _scrape_today_line_usage(self):
        """
        Scrapes line usage for today (the first table on the page)
        :return: List with 64 items - one for each 15 segment when pool is open. 
                 Each item contains number that represents number of reserved lines in given 15 minute slot.
        """
        r = urllib.request.urlopen(self.line_usage_url).read()
        soup = BeautifulSoup(r, 'html.parser')
        table = soup.find('table', attrs={'class':'pooltable'})
        table_rows = table.find_all('tr')

        time_slots = [0]*64
        i = 0
        for i in range(1,9): 
            row = table_rows[i]
            slot = 0
            row_str = '|'
            cols = row.find_all('td')
            for col in cols:
                if len(col) == 3:
                    if 'colspan' in col.attrs:
                        row_str += ' * | * |'
                        time_slots[slot] += 1
                        time_slots[slot+1] += 1
                        slot += 2
                    else:
                        row_str += ' * |'
                        time_slots[slot] += 1
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

    def _get_current_line_usage(self):
        """
        Returns current number of reserved line in the pool
        :return: Integer with number of currently reserved lines
        """
        now = datetime.datetime.now()
        if now.hour < 6 or now.hour >= 22:
            return 0
        else:
            slots = self._scrape_today_line_usage()

            slot_id = (now.hour-6)*4
            if now.minute >= 45:
                slot_id += 3
            elif now.minute >= 30:
                slot_id += 2
            elif now.minute >= 15:
                slot_id += 1
            return slots[slot_id]
    
    def get_current_occupancy(self):
        """
        Scrapes current occupancy of Sutka swimming pool.
        :return: Dictionary with following keys:
                    pool: Number of people currently in the pool
                    park: Number of people currently in the park
                    percentage: Occupancy percentage
                    lines: Number of currently reserved swimming lines
        """
        data = self._scrape_current_occupancy_data()
        if data is None:
            return None
        lines = self._get_current_line_usage()
        occupancy = {
            'pool': int(data[2].get_text()),
            'park': int(data[3].get_text()),
            'lines': int(lines),
            'percentage': int(data[1].get_text()[:-1])
        }

        return occupancy

if __name__=='__main__':
    os = OccupancyScraper()
    print(os.get_current_occupancy())
