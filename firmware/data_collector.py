import json
import time
from datetime import datetime

from serial import Serial

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------

SENSOR_NODE_PORT = 'COM4'
NUMBER_OF_SAMPLES = 5 * 64  # 5 periods with 64 dots
HOME_APPLIANCE = "Ventilador"
NUMBER_OF_READINGS = 100  # number of readings taken at the sensor node

registers = []

# ----------------------------------------------------------------------------------------------------------------------
#  Data acquisition
# ----------------------------------------------------------------------------------------------------------------------

'''
Array Data Protocol:
#  index     type       data description
# 0 - 319    int     electric current signal
#   320     float    electric real power
#   321     float    electric apparent power
#   322     float    electric power factor
'''


def read_all_features_from_serial():
    connection_serial = Serial(SENSOR_NODE_PORT, baudrate=115200, timeout=1)
    time.sleep(1.8)

    _current_signal_temp = []

    # start the acquisition
    connection_serial.write(b't')

    for i in range(0, NUMBER_OF_SAMPLES):
        d = connection_serial.readline()
        _current_signal_temp.append(float(d))

    _real_power = float(connection_serial.readline())
    _apparent_power = float(connection_serial.readline())
    _power_factor = float(connection_serial.readline())

    connection_serial.close()

    return _power_factor, _real_power, _apparent_power, _current_signal_temp


# ------------------------------------------------------------------------------------------------------------------
# Save time series in JSON format
# ------------------------------------------------------------------------------------------------------------------
for k in range(NUMBER_OF_READINGS):

    power_factor, real_power, apparent_power, current_signal_temp = read_all_features_from_serial()

    registers.append({
        'home_appliance': HOME_APPLIANCE,
        'apparent_power': apparent_power,
        'real_power': real_power,
        'power_factor': power_factor,
        'current_signal_temp': current_signal_temp
    })

    print(HOME_APPLIANCE, k + 1, 'PA:', apparent_power,
          'PR', real_power, " FP: ", power_factor)

# ----------------------------------------------------------------------------------------------------------------------

now = datetime.now()
file_name = HOME_APPLIANCE + '-' + str(now.timestamp()) + '.json'
with open('time-series/' + file_name, 'w') as f:
    json.dump(registers, f)

print("Time Series Saved")
