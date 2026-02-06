"""
AgroSmart ESP32 Serial Bridge
==============================
Reads sensor data from ESP32 via USB serial and sends to Flask server.
Also forwards pump commands from server to ESP32.

Usage:
    python serial_bridge.py          # Auto-detect COM port
    python serial_bridge.py COM3     # Specify COM port
    python serial_bridge.py /dev/ttyUSB0  # Linux/Mac
"""

import serial
import serial.tools.list_ports
import requests
import json
import time
import sys
import threading
from datetime import datetime

# Configuration
SERVER_URL = "http://127.0.0.1:5000"
BAUD_RATE = 115200
READ_TIMEOUT = 2  # seconds

# Global state
pending_command = None
running = True


def find_esp32_port():
    """Auto-detect ESP32 COM port."""
    ports = serial.tools.list_ports.comports()
    
    print("\n📡 Scanning for ESP32...")
    print("-" * 40)
    
    for port in ports:
        print(f"  Found: {port.device} - {port.description}")
        
        # Common ESP32 identifiers
        if any(x in port.description.lower() for x in ['cp210', 'ch340', 'usb serial', 'uart', 'esp32', 'silicon labs']):
            print(f"  ✓ Likely ESP32: {port.device}")
            return port.device
    
    if ports:
        # Return first available port if no ESP32-specific found
        print(f"  Using first available: {ports[0].device}")
        return ports[0].device
    
    return None


def send_to_server(data):
    """Send sensor data to Flask server."""
    global pending_command
    
    try:
        response = requests.post(
            f"{SERVER_URL}/api/esp32/data",
            json=data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check for pump command from server
            if "pump_command" in result:
                pending_command = result["pump_command"]
                print(f"  📥 Server command: {pending_command}")
            
            return True
        else:
            print(f"  ⚠ Server error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ⚠ Cannot connect to server - is it running?")
        return False
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return False


def send_command_to_esp32(ser, command):
    """Send pump command to ESP32."""
    try:
        # Send as plain text command (ESP32 expects PUMP_ON, PUMP_OFF, or AUTO)
        if command == "ON":
            cmd = "PUMP_ON\n"
        elif command == "OFF":
            cmd = "PUMP_OFF\n"
        elif command == "AUTO":
            cmd = "AUTO\n"
        else:
            print(f"  ⚠ Unknown command: {command}")
            return
        
        ser.write(cmd.encode())
        ser.flush()
        print(f"  📤 Sent to ESP32: {cmd.strip()}")
    except Exception as e:
        print(f"  ⚠ Failed to send command: {e}")


def read_serial_data(ser):
    """Read and parse JSON data from serial."""
    try:
        line = ser.readline().decode('utf-8').strip()
        
        if not line:
            return None
        
        # Try to parse as JSON
        try:
            data = json.loads(line)
            return data
        except json.JSONDecodeError:
            # Not JSON, might be debug message
            if line and not line.startswith('{'):
                print(f"  ESP32: {line}")
            return None
            
    except Exception as e:
        return None


def print_sensor_data(data):
    """Pretty print sensor readings."""
    print("\n" + "=" * 50)
    print(f"  🌱 AgroSmart Live Sensor Data")
    print(f"  ⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    # Handle values that might be strings or numbers
    moisture = data.get('soil_moisture', 0)
    temp = data.get('temperature', 0)
    humid = data.get('humidity', 0)
    raw = data.get('raw_moisture', 0)
    
    try:
        print(f"  💧 Soil Moisture: {float(moisture):.1f}%")
        print(f"  🌡️  Temperature:   {float(temp):.1f}°C")
        print(f"  💨 Humidity:      {float(humid):.1f}%")
        print(f"  🔧 Raw ADC:       {raw}")
    except (ValueError, TypeError):
        print(f"  💧 Soil Moisture: {moisture}%")
        print(f"  🌡️  Temperature:   {temp}°C")
        print(f"  💨 Humidity:      {humid}%")
        print(f"  🔧 Raw ADC:       {raw}")
    
    pump_status = "🟢 RUNNING" if data.get('pump_running') else "⚪ OFF"
    print(f"  🚰 Pump:          {pump_status}")
    
    if data.get('pump_running') and data.get('pump_runtime', 0) > 0:
        print(f"  ⏱️  Runtime:       {data.get('pump_runtime')}s")
    
    mode = "🤖 AUTO" if data.get('auto_mode') else "🎮 MANUAL"
    print(f"  📡 Mode:          {mode}")
    print("=" * 50)


def main():
    global pending_command, running
    
    print("\n" + "=" * 50)
    print("  🌿 AgroSmart ESP32 Serial Bridge")
    print("  Connecting ESP32 to Dashboard via USB")
    print("=" * 50)
    
    # Determine COM port
    if len(sys.argv) > 1:
        port = sys.argv[1]
        print(f"\n📌 Using specified port: {port}")
    else:
        port = find_esp32_port()
        if not port:
            print("\n❌ No serial ports found!")
            print("   Make sure ESP32 is connected via USB.")
            print("\n   Usage: python serial_bridge.py COM3")
            sys.exit(1)
    
    # Connect to serial port
    print(f"\n🔌 Connecting to {port} at {BAUD_RATE} baud...")
    
    try:
        # Open without DTR/RTS to prevent ESP32 auto-reset
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = BAUD_RATE
        ser.timeout = READ_TIMEOUT
        ser.dtr = False
        ser.rts = False
        ser.open()
        time.sleep(1)  # Wait for connection to stabilize
        ser.reset_input_buffer()  # Clear any garbage in the buffer
        ser.reset_output_buffer()
        time.sleep(0.5)
        print(f"✓ Connected to {port} (DTR/RTS disabled)")
    except serial.SerialException as e:
        print(f"\n❌ Failed to open {port}: {e}")
        print("\n   Possible solutions:")
        print("   1. Check if ESP32 is connected")
        print("   2. Close Arduino Serial Monitor if open")
        print("   3. Try a different COM port")
        sys.exit(1)
    
    print(f"\n🌐 Sending data to: {SERVER_URL}")
    print("   Press Ctrl+C to stop\n")
    print("-" * 50)
    
    # Main loop
    try:
        while running:
            # Read data from ESP32
            data = read_serial_data(ser)
            
            if data and "device_id" in data:
                # Print sensor data
                print_sensor_data(data)
                
                # Send to server
                print("  📤 Sending to server...", end=" ")
                if send_to_server(data):
                    print("✓")
                else:
                    print("✗")
                
                # Send pending command to ESP32
                if pending_command:
                    send_command_to_esp32(ser, pending_command)
                    pending_command = None
            
            # Small delay to prevent CPU overload
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        running = False
    finally:
        ser.close()
        print("✓ Serial port closed")


if __name__ == "__main__":
    main()
