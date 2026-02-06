# AgroSmart ESP32 Setup Guide (USB Serial Mode)

## Overview

This setup uses **USB cable** to connect ESP32 to your computer. A Python script reads sensor data from the serial port and sends it to the dashboard.

```
[ESP32 + Sensors] --USB Cable--> [Computer] --HTTP--> [Flask Server] --> [Dashboard]
```

## Hardware Requirements

| Component | Model | Connection |
|-----------|-------|------------|
| Microcontroller | ESP32 DevKit | USB to Computer |
| Temperature & Humidity | DHT11 | Data → GPIO 4 |
| Soil Moisture Sensor | Capacitive v1.2/v2.0 | Analog → GPIO 34 |
| Relay Module | 5V 1-Channel | Signal → GPIO 26 |
| Water Pump | 5V/12V DC Mini Pump | Via Relay |

## Wiring Diagram

```
ESP32 DevKit V1
┌─────────────────────────────────────┐
│                                     │
│  USB ───────────── Computer         │
│                                     │
│  3.3V ──────────── DHT11 VCC        │
│  GND  ──────────── DHT11 GND        │
│  GPIO4 ─────────── DHT11 DATA       │
│                                     │
│  3.3V ──────────── Moisture VCC     │
│  GND  ──────────── Moisture GND     │
│  GPIO34 ────────── Moisture AOUT    │
│                                     │
│  5V (VIN) ──────── Relay VCC        │
│  GND  ──────────── Relay GND        │
│  GPIO26 ────────── Relay IN         │
│                                     │
│  Relay COM ─────── Pump (+)         │
│  Relay NO ──────── Power Supply (+) │
│  Power Supply (-) ─ Pump (-)        │
│                                     │
└─────────────────────────────────────┘
```

## Quick Start

### Step 1: Upload Code to ESP32

1. Open Arduino IDE
2. Open `agrosmart_esp32.ino`
3. Select Board: **Tools → Board → ESP32 Dev Module**
4. Select Port: **Tools → Port → COMx** (your ESP32)
5. Click **Upload**

### Step 2: Install Python Dependencies

```bash
cd agrotech/esp32
pip install pyserial requests
```

### Step 3: Start the Flask Server

```bash
cd agrotech
python farm_agent_server.py
```

### Step 4: Run the Serial Bridge

```bash
cd agrotech/esp32
python serial_bridge.py
```

Or specify COM port directly:
```bash
python serial_bridge.py COM3        # Windows
python serial_bridge.py /dev/ttyUSB0  # Linux
```

### Step 5: View Dashboard

Open http://localhost:8080 - you'll see live sensor data!

## Arduino IDE Setup

### 1. Install Arduino IDE
Download from: https://www.arduino.cc/en/software

### 2. Add ESP32 Board Support
1. Open Arduino IDE → File → Preferences
2. Add to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Tools → Board → Boards Manager → Search "ESP32" → Install

### 3. Install Required Libraries
Sketch → Include Library → Manage Libraries:
- **DHT sensor library** by Adafruit
- **ArduinoJson** by Benoit Blanchon
- **Adafruit Unified Sensor**

## Calibrating Moisture Sensor

1. Place sensor in **dry air** → Note the serial output value (e.g., 3500)
2. Place sensor in **water** → Note the value (e.g., 1500)
3. Update in `agrosmart_esp32.ino`:

```cpp
#define MOISTURE_DRY 3500   // Your dry value
#define MOISTURE_WET 1500   // Your wet value
```

## Serial Bridge Output

When running `serial_bridge.py`, you'll see:

```
==================================================
  🌿 AgroSmart ESP32 Serial Bridge
  Connecting ESP32 to Dashboard via USB
==================================================

📡 Scanning for ESP32...
  Found: COM3 - Silicon Labs CP210x USB to UART Bridge
  ✓ Likely ESP32: COM3

🔌 Connecting to COM3 at 115200 baud...
✓ Connected to COM3

🌐 Sending data to: http://127.0.0.1:5000
   Press Ctrl+C to stop

==================================================
  🌱 AgroSmart Live Sensor Data
  ⏰ 14:32:15
==================================================
  💧 Soil Moisture: 45.3%
  🌡️  Temperature:   28.5°C
  💨 Humidity:      65.2%
  🔧 Raw ADC:       2450
  🚰 Pump:          ⚪ OFF
  📡 Mode:          🤖 AUTO
==================================================
  📤 Sending to server... ✓
```

## Pump Control from Dashboard

1. Click **ON** - Turns pump on (manual mode)
2. Click **OFF** - Turns pump off (manual mode)
3. Click **AUTO** - Returns to automatic control

Commands flow: Dashboard → Flask Server → Serial Bridge → ESP32

## Troubleshooting

### "No serial ports found"
- Check USB cable connection
- Install ESP32 USB drivers (CP210x or CH340)
- Try different USB port

### "Failed to open COMx"
- Close Arduino Serial Monitor (only one app can use the port)
- Check Device Manager for correct COM port number
- Run as Administrator if permission denied

### Wrong moisture readings
- Recalibrate MOISTURE_DRY and MOISTURE_WET values
- Ensure sensor is in soil, not floating
- Check wiring to GPIO 34

### DHT11 errors
- Add 10kΩ resistor between DATA and VCC
- Check wiring to GPIO 4
- Ensure DHT11 is powered (3.3V)

### Pump won't respond
- Check relay wiring
- Verify GPIO 26 connection
- Test relay with LED first

## Files

| File | Description |
|------|-------------|
| `agrosmart_esp32.ino` | Arduino code for ESP32 |
| `serial_bridge.py` | Python script to bridge serial→server |
| `README.md` | This guide |

## LED Indicators

| LED State | Meaning |
|-----------|---------|
| 3 Quick Blinks | Startup complete |
| OFF | Normal operation, pump off |
| Blinking | Pump is running |
