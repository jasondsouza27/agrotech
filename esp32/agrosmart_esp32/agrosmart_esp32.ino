#include <AppInsights.h>
#include <RMaker.h>
#include <RMakerDevice.h>
#include <RMakerNode.h>
#include <RMakerParam.h>
#include <RMakerQR.h>
#include <RMakerType.h>
#include <RMakerUtils.h>

/*
 * AgroSmart ESP32 - Live Sensor Module (USB Serial Mode)
 * =======================================================
 * 
 * This version sends data over USB serial cable to the computer.
 * A Python script reads the serial data and sends it to the dashboard.
 * 
 * Hardware Connections:
 * - DHT11: Data pin -> GPIO 4
 * - Capacitive Moisture Sensor: Analog out -> GPIO 34 (ADC1_CH6)
 * - Relay Module (Pump): Signal -> GPIO 26
 * - LED Indicator: GPIO 2 (onboard LED)
 * 
 * Connection: USB Cable to Computer
 */

#include <ArduinoJson.h>
#include <DHT.h>

// ==================== CONFIGURATION ====================

// Sensor Pins
#define DHT_PIN 4              // DHT11 data pin
#define DHT_TYPE DHT11         // DHT sensor type
#define MOISTURE_PIN 34        // Capacitive moisture sensor (ADC)
#define RELAY_PIN 26           // Relay for pump control
#define LED_PIN 2              // Onboard LED indicator

// Moisture Calibration Values (adjust based on your sensor)
// Dry soil = higher value, Wet soil = lower value
#define MOISTURE_DRY 3500      // ADC value when sensor is in dry air
#define MOISTURE_WET 1500      // ADC value when sensor is in water

// Pump Control Thresholds
#define MOISTURE_THRESHOLD_LOW 30   // Start pump below this %
#define MOISTURE_THRESHOLD_HIGH 60  // Stop pump above this %

// Timing
#define SENSOR_READ_INTERVAL 2000   // Read sensors every 2 seconds
#define SERIAL_BAUD_RATE 115200     // Serial communication speed

// ==================== GLOBAL VARIABLES ====================

DHT dht(DHT_PIN, DHT_TYPE);

// Sensor readings
float temperature = 0;
float humidity = 0;
float soilMoisture = 0;
int rawMoistureValue = 0;

// Pump state
bool pumpRunning = false;
bool autoMode = true;           // Automatic pump control based on moisture
unsigned long pumpStartTime = 0;
unsigned long pumpRuntime = 0;  // Runtime in seconds

// Timing
unsigned long lastSensorRead = 0;

// Device ID
String deviceId = "ESP32_USB";

// ==================== SETUP ====================

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  
  // Wait for serial connection
  while (!Serial) {
    delay(10);
  }
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(MOISTURE_PIN, INPUT);
  
  // Ensure pump is OFF at startup
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  
  // Initialize DHT sensor
  dht.begin();
  
  // Generate device ID from chip ID
  deviceId = "ESP32_" + String((uint32_t)ESP.getEfuseMac(), HEX);
  
  // Signal ready with LED blinks
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  
  // Send startup message
  Serial.println("{\"status\":\"ready\",\"device_id\":\"" + deviceId + "\"}");
}

// ==================== MAIN LOOP ====================

void loop() {
  // Check for incoming commands from computer
  checkSerialCommands();
  
  // Read sensors at interval
  if (millis() - lastSensorRead >= SENSOR_READ_INTERVAL) {
    lastSensorRead = millis();
    
    // Read all sensors
    readSensors();
    
    // Auto pump control based on moisture
    if (autoMode) {
      controlPump();
    }
    
    // Send data as JSON over serial
    sendSensorData();
  }

  // Blink LED when pump is running
  if (pumpRunning) {
    digitalWrite(LED_PIN, (millis() / 500) % 2);
  } else {
    digitalWrite(LED_PIN, LOW);
  }
}

// ==================== SERIAL COMMANDS ====================

void checkSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Parse JSON command
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, command);
    
    if (!error) {
      if (doc.containsKey("pump")) {
        String pumpCmd = doc["pump"].as<String>();
        
        if (pumpCmd == "ON") {
          startPump();
          autoMode = false;
          Serial.println("{\"ack\":\"pump_on\"}");
        } else if (pumpCmd == "OFF") {
          stopPump();
          autoMode = false;
          Serial.println("{\"ack\":\"pump_off\"}");
        } else if (pumpCmd == "AUTO") {
          autoMode = true;
          Serial.println("{\"ack\":\"auto_mode\"}");
        }
      }
    }
  }
}

// ==================== SENSOR READING ====================

void readSensors() {
  // Read DHT11 (Temperature & Humidity)
  float newTemp = dht.readTemperature();
  float newHum = dht.readHumidity();
  
  // Validate DHT readings
  if (!isnan(newTemp) && !isnan(newHum)) {
    temperature = newTemp;
    humidity = newHum;
  }
  
  // Read Capacitive Moisture Sensor
  rawMoistureValue = analogRead(MOISTURE_PIN);
  
  // Convert to percentage (0-100%)
  soilMoisture = map(rawMoistureValue, MOISTURE_DRY, MOISTURE_WET, 0, 100);
  soilMoisture = constrain(soilMoisture, 0, 100);
  
  // Update pump runtime if running
  if (pumpRunning) {
    pumpRuntime = (millis() - pumpStartTime) / 1000;
  }
}

void sendSensorData() {
  // Create JSON document
  StaticJsonDocument<300> doc;
  
  doc["device_id"] = deviceId;
  doc["soil_moisture"] = round(soilMoisture * 10) / 10.0;  // 1 decimal
  doc["temperature"] = round(temperature * 10) / 10.0;
  doc["humidity"] = round(humidity * 10) / 10.0;
  doc["raw_moisture"] = rawMoistureValue;
  doc["pump_running"] = pumpRunning;
  doc["pump_runtime"] = pumpRuntime;
  doc["auto_mode"] = autoMode;
  doc["uptime"] = millis() / 1000;
  
  // Send as single line JSON
  serializeJson(doc, Serial);
  Serial.println();  // Newline to mark end of message
}

// ==================== PUMP CONTROL ====================

void controlPump() {
  // Safety check: Stop pump if running too long (30 minutes max)
  if (pumpRunning && pumpRuntime > 1800) {
    stopPump();
    return;
  }
  
  // Start pump if moisture is too low
  if (!pumpRunning && soilMoisture < MOISTURE_THRESHOLD_LOW) {
    startPump();
  }
  
  // Stop pump if moisture is adequate
  if (pumpRunning && soilMoisture > MOISTURE_THRESHOLD_HIGH) {
    stopPump();
  }
}

void startPump() {
  digitalWrite(RELAY_PIN, HIGH);
  pumpRunning = true;
  pumpStartTime = millis();
  pumpRuntime = 0;
}

void stopPump() {
  digitalWrite(RELAY_PIN, LOW);
  pumpRunning = false;
  pumpRuntime = 0;
}
