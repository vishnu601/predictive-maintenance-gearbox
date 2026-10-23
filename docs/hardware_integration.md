# Hardware Integration & Serial Sensor Protocol

This document specifies the hardware interface for connecting real-time physical vibration sensors (accelerometers) and microcontrollers to the monitoring platform.

---

## 1. Hardware Architecture Overview

```
[ Gearbox Casing ]
       │
   (Stud / Magnetic Mount)
       │
[ Accelerometer (e.g. ADXL355, MPU6050, PCB Piezotronics) ]
       │ (Analog / SPI / I2C)
[ Microcontroller / DAQ ADC (Arduino / STM32 / ESP32) ]
       │ (USB-UART Virtual COM Port)
[ Predictive Maintenance Platform (SerialSensorSource) ]
```

---

## 2. Serial Communication Protocol

- **Physical Layer:** USB-to-UART Serial (e.g., FTDI, CP2102, CH340).
- **Default Baud Rate:** 115,200 bps (configurable to 9,600, 57,600, or 230,400).
- **Configuration:** 8 Data Bits, No Parity, 1 Stop Bit (8-N-1), No Hardware Flow Control.
- **Framing Convention:** ASCII Line-Terminated (`\n` or `\r\n`).

### Supported Packet Formats

#### Option A: Timestamped CSV (Recommended)
```text
<timestamp_ms>,<acceleration_g>\n
```
*Example:*
```text
124500,1.432
124501,1.389
124502,-0.921
```

#### Option B: Raw Float Stream
```text
<acceleration_g>\n
```
*Example:*
```text
0.412
0.589
-1.204
```

---

## 3. Sample Microcontroller Firmware (Arduino / C++)

Below is a reference sketch reading an analog accelerometer at 5,000 Hz using timer interrupts and transmitting framed samples over USB serial:

```cpp
#include <Arduino.h>

const int VIBRATION_PIN = A0;
const float VOLTS_PER_G = 0.330; // Sensor sensitivity (e.g. 330 mV/g)
const float ZERO_G_VOLTS = 1.65; // Zero-g bias voltage

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    pinMode(VIBRATION_PIN, INPUT);
}

void loop() {
    // Read ADC (0 to 1023 for 10-bit ADC at 3.3V)
    int raw_adc = analogRead(VIBRATION_PIN);
    float voltage = (raw_adc / 1023.0) * 3.3;
    float acceleration_g = (voltage - ZERO_G_VOLTS) / VOLTS_PER_G;

    // Transmit framed packet
    Serial.print(millis());
    Serial.print(",");
    Serial.println(acceleration_g, 4);

    // Maintain nominal sampling delay (e.g., 200 microseconds for 5 kHz)
    delayMicroseconds(200);
}
```

---

## 4. Mounting & Signal Integrity Guidelines
1. **Sensor Placement:** Mount the accelerometer as close to the gear mesh zone as possible, on a flat, rigid section of the gearbox casing.
2. **Mounting Method:** Prefer stud mounting or high-stiffness magnetic bases. Avoid flexible tape or loose clamps, which act as mechanical low-pass filters and dampen high-frequency gear-mesh harmonics.
3. **Cable Strain Relief:** Secure the coaxial cable to prevent cable whip and triboelectric noise artifacts.
