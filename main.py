from datetime import datetime

# make CSV file name from these params
time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
filename = f'logfile_{time_stamp}.csv'
logfile = open(filename, 'w')
# logfile.write('time, alt, fuel, vs, hs, distance, angle, engine\n')
# self.logfile.write(
#                 f'{self.time}, {self.altitude}, {self.fuel}, {self.vertical_speed}, {self.horizontal_speed}, {self.distance}, {self.angle}, {int(self.engine_is_on)}\n')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
