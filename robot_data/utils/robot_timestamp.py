from datetime import datetime
from datetime import timezone
import time

class RobotTimestampsDecoder():
    def __init__(self, timestamp) -> None:
        self.timestamp_ns = 0
        self.timestamp_sec = 0
        if isinstance(timestamp, str) or isinstance(timestamp, float) or isinstance(timestamp, int):
            self.t = str(timestamp)
            tmp_t = self.t.split('.')
            if len(tmp_t[0]) < 10:
                # timestamp(10e11) == 2286-11-20  
                print(f'{timestamp} is not a valid timestamp')
                raise ValueError
            elif len(tmp_t[0]) == 10:
                self.timestamp_sec = tmp_t[0]
            else:
                self.timestamp_sec = tmp_t[0][:10]
                self.timestamp_ns = tmp_t[0][10:]
            if len(tmp_t) == 2:
                self.timestamp_ns = tmp_t[1]
        elif (isinstance(timestamp, tuple) or isinstance(timestamp, list)) and len(timestamp)==2:
            if len(str(timestamp[0])) != 10:
                print(f'{timestamp} is not a valid timestamp')
                raise ValueError
            self.timestamp_ns = str(timestamp[0])
            self.timestamp_sec = str(timestamp[1])
        else:
            raise NotImplementedError

        
        self.t = float(f'{self.timestamp_sec}.{self.timestamp_ns}')
        self.format_t = datetime.utcfromtimestamp(self.t)
        
        
        
    def format_timestamp(self, format_patten):
        return self.format_t.strftime(format_patten)
    
    def date(self):
        return self.format_timestamp('%Y-%m-%d')
    
    def date_and_time(self):
        return self.format_timestamp('%Y-%m-%d-%H-%M-%S')
    
    def get_raw_ts(self):
        return self.t
    
    def hour(self):
        return self.format_t.hour

    def month(self):
        return self.format_t.month
        
        
class RobotTimestampsIncoder():
    def __init__(self) -> None:
        self.timestamp_ns = 0
        self.timestamp_sec = 0
        
        self.t = float(f'{self.timestamp_sec}.{self.timestamp_ns}')
        self.format_t = datetime.utcfromtimestamp(self.t)
        
    def set_current_timestamp(self):
        current_timestamp = "{:.7f}".format(time.time())#datetime.now(timezone.utc)
        self.timestamp_ns = current_timestamp.split(".")[1]
        self.timestamp_sec = current_timestamp.split(".")[0]
        return f'{self.timestamp_sec}.{self.timestamp_ns}'
        
    def format_timestamp(self, format_patten):
        return self.format_t.strftime(format_patten)
    
    def date(self):
        return self.format_timestamp('%Y-%m-%d')
    
    def date_and_time(self):
        return self.format_timestamp('%Y-%m-%d-%H-%M-%S')
    
    def get_raw_ts(self):
        return self.t
    
    def hour(self):
        return self.format_t.hour

    def month(self):
        return self.format_t.month
        
if __name__ == "__main__":
    robot_timestamps_incoder = RobotTimestampsIncoder()
    for i in range(100):
        print(robot_timestamps_incoder.set_current_timestamp())
