import csv
import json

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------

NUMBER_OF_SAMPLES = 5 * 64
N_PERIODS = 7
dt = (N_PERIODS / 60) / NUMBER_OF_SAMPLES
time_series = []

# ----------------------------------------------------------------------------------------------------------------------
# Read the data mass
# ----------------------------------------------------------------------------------------------------------------------

for json_data in json.load(open('data/timeseries.json')):
    time_series.append(json_data)

# ----------------------------------------------------------------------------------------------------------------------
# Create dataset
# ----------------------------------------------------------------------------------------------------------------------


def _normalize_signal_limits(signal):
    return signal[1:len(signal) // 2]


def _generate_all_frequencies():
    return _normalize_signal_limits(np.fft.fftfreq(NUMBER_OF_SAMPLES, dt))


def generate_frequencies_harmonics_labeled():
    freq = _generate_all_frequencies()
    res = []
    for i in range(N_PERIODS - 1, len(freq), N_PERIODS):
        res.append(str(int(freq[i])) + "Hz")
    return res


def extract_harmonic_components(wave_form):
    # normalized Fourier coefficients
    signal_freq = _normalize_signal_limits(abs(np.fft.fft(wave_form) / NUMBER_OF_SAMPLES))
    _harmonics = []
    for i in range(N_PERIODS - 1, len(signal_freq), N_PERIODS):
        _harmonics.append(signal_freq[i])
    return _harmonics


with open('data/dataset.csv', 'w+', newline='\n', encoding='utf-8') as file:
    writer = csv.writer(file)

    line = ["home_appliance", "real_power"] + generate_frequencies_harmonics_labeled()
    writer.writerow(line)

    for t in time_series:
        home_appliance = t['home_appliance']
        real_power = t['real_power']
        harmonics = extract_harmonic_components(t['current_signal_temp'])
        line = [home_appliance, real_power] + harmonics
        writer.writerow(line)

print("dataset created")
