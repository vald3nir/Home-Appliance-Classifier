#include <Arduino.h>

// =========================================================================================
// Definitions
// =========================================================================================

#define I_VCC 3.3f // supply voltage of the current sensor
#define V_VCC 5.0f // supply voltage of the voltage sensor

#define NUMBER_OF_SAMPLES 5 * 64 // 5 periods with 64 samples each

#define ADC_BITS 10
#define ADC_COUNTS (1 << ADC_BITS)

#define CURRENT_CONVERSION_CONSTANT (I_VCC / ADC_COUNTS) * 0.15f
#define VOLTAGE_CONVERSION_CONSTANT (V_VCC / ADC_COUNTS) * 7.75f

#define INPUT_PIN_VOLTAGE A1
#define INPUT_PIN_CURRENT A0

#define ACTIVATION_CHARACTER 116 //'t'

// =========================================================================================

double offsetI = 335.0; // Low-pass filter output [ADC_COUNTS >> 1]
double offsetV = 510.0; // Low-pass filter output [ADC_COUNTS >> 1]

double sumI = 0.0f, sumV = 0.0f, sumP = 0.0f;

double filteredI, filteredV;
double apparent_power = 66.71f;
double real_power = 37.25f;
double voltage = 220.0f;
double current = 0.30f;
double power_factor = real_power / apparent_power;

unsigned int n = 0;

// Buffer para armazenar as amostras da forma de onda
int currentWaveForm[NUMBER_OF_SAMPLES] = {336, 336, 336, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 335, 352, 369, 353, 347, 339, 330, 330, 329, 330, 330, 329, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 330, 332, 296, 306, 315, 321, 333, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 334, 335, 335, 340, 369, 355, 349, 340, 331, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 331, 300, 304, 312, 320, 330, 335, 336, 335, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 371, 357, 350, 340, 331, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 331, 331, 305, 303, 311, 318, 329, 335, 335, 335, 335, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 334, 334, 371, 357, 350, 342, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 332, 312, 301, 312, 317, 329, 335, 335, 335, 335, 335, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 334, 335, 334, 371, 359, 351, 343, 331, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 332, 331, 322, 298, 310, 318, 327, 335, 335, 335, 335, 335, 336, 336, 335, 335, 335, 335, 335, 335, 335, 334, 335, 334, 334, 371, 359, 351, 344, 332, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 330, 330, 328, 297, 309, 315, 324, 335, 336, 336, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 335, 333, 364, 360, 352, 345, 333, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 331, 331, 333, 295, 309, 314, 325, 335, 335, 336, 335, 335, 336, 335, 335};
int voltageWaveForm[NUMBER_OF_SAMPLES] = {336, 336, 336, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 335, 352, 369, 353, 347, 339, 330, 330, 329, 330, 330, 329, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 330, 332, 296, 306, 315, 321, 333, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 334, 335, 335, 340, 369, 355, 349, 340, 331, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 331, 300, 304, 312, 320, 330, 335, 336, 335, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 371, 357, 350, 340, 331, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 331, 331, 305, 303, 311, 318, 329, 335, 335, 335, 335, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 334, 334, 371, 357, 350, 342, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 332, 312, 301, 312, 317, 329, 335, 335, 335, 335, 335, 335, 336, 335, 335, 335, 335, 335, 335, 335, 335, 334, 335, 334, 371, 359, 351, 343, 331, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 332, 331, 322, 298, 310, 318, 327, 335, 335, 335, 335, 335, 336, 336, 335, 335, 335, 335, 335, 335, 335, 334, 335, 334, 334, 371, 359, 351, 344, 332, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 330, 330, 328, 297, 309, 315, 324, 335, 336, 336, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 334, 334, 335, 333, 364, 360, 352, 345, 333, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 331, 331, 333, 295, 309, 314, 325, 335, 335, 336, 335, 335, 336, 335, 335};

// =========================================================================================

void process_wave_form()
{
    for (n = 0; n < NUMBER_OF_SAMPLES; n++)
    {
        currentWaveForm[n] = analogRead(INPUT_PIN_CURRENT);
        voltageWaveForm[n] = analogRead(INPUT_PIN_VOLTAGE);
        delayMicroseconds(148);
    }

    for (n = 0; n < NUMBER_OF_SAMPLES; n++) // Current RMS and Voltage RMS
    {
        offsetI = (offsetI + (currentWaveForm[n] - offsetI) / 1024);
        offsetV = offsetV + ((voltageWaveForm[n] - offsetV) / 1024);

        filteredI = currentWaveForm[n] - offsetI;
        filteredV = voltageWaveForm[n] - offsetV;

        sumI += (filteredI * filteredI);
        sumV += (filteredV * filteredV);
        sumP += (filteredI * filteredV);
    }

    real_power = VOLTAGE_CONVERSION_CONSTANT * CURRENT_CONVERSION_CONSTANT * (sumP / NUMBER_OF_SAMPLES);
    voltage = (VOLTAGE_CONVERSION_CONSTANT * sqrt(sumV / NUMBER_OF_SAMPLES));
    current = (CURRENT_CONVERSION_CONSTANT * sqrt(sumI / NUMBER_OF_SAMPLES));
    apparent_power = voltage * current;
    power_factor = (apparent_power == 0) ? 0.0f : (real_power / apparent_power);

    sumP = 0;
    sumI = 0;
    sumV = 0;
}

void setup()
{
    Serial.begin(115200);
}

void loop()
{
    if (Serial.available() > 0 && Serial.read() == ACTIVATION_CHARACTER)
    {
        process_wave_form();

        /*
        Array Data Protocol :
              index    type       data description
             0 - 319   int     electric current signal
               320     float   electric real power
               321     float   electric apparent power
               322     float   electric power factor
        */
        for (n = 0; n < NUMBER_OF_SAMPLES; n++)
        {
            Serial.println(currentWaveForm[n]);
        }
        Serial.println(String(real_power, 6));
        Serial.println(String(apparent_power, 6));
        Serial.println(String(power_factor, 6));
    }
}
