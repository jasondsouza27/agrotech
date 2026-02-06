import { useState, useEffect } from "react";
import { 
  Wifi, WifiOff, Droplets, Thermometer, Wind, 
  Activity, Power, RefreshCw, Cpu, Signal
} from "lucide-react";

interface SensorData {
  device_id: string;
  soil_moisture: number;
  temperature: number;
  humidity: number;
  pump_running: boolean;
  pump_runtime: number;
  auto_mode: boolean;
  wifi_rssi?: number;
  uptime?: number;
  last_update: string;
  is_simulated?: boolean;
  is_online?: boolean;
}

export function LiveSensorData() {
  const [sensorData, setSensorData] = useState<SensorData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchSensorData = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/api/esp32/sensors");
      
      if (response.ok) {
        const data = await response.json();
        // Get first device's data
        if (data.devices && data.devices.length > 0) {
          setSensorData(data.devices[0]);
          setError(null);
        } else {
          setSensorData(null);
        }
      } else {
        throw new Error("Failed to fetch sensor data");
      }
    } catch (err) {
      console.error("Sensor fetch error:", err);
      setError("Cannot connect to server");
      // Show simulated data on error
      setSensorData({
        device_id: "DEMO",
        soil_moisture: 52,
        temperature: 28,
        humidity: 65,
        pump_running: false,
        pump_runtime: 0,
        auto_mode: true,
        last_update: new Date().toISOString(),
        is_simulated: true
      });
    }
    setLoading(false);
    setLastRefresh(new Date());
  };

  useEffect(() => {
    fetchSensorData();
    // Auto-refresh every 250ms for ultra-fast real-time updates
    const interval = setInterval(fetchSensorData, 250);
    return () => clearInterval(interval);
  }, []);

  const sendPumpCommand = async (command: "ON" | "OFF" | "AUTO") => {
    if (!sensorData) return;
    
    try {
      const response = await fetch("http://127.0.0.1:5000/api/esp32/pump", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: sensorData.device_id,
          command: command
        })
      });
      
      if (response.ok) {
        // Refresh data after command
        setTimeout(fetchSensorData, 1000);
      }
    } catch (err) {
      console.error("Pump command error:", err);
    }
  };

  const getMoistureStatus = (moisture: number) => {
    if (moisture < 30) return { color: "text-red-500", bg: "bg-red-500/20", label: "Dry" };
    if (moisture < 50) return { color: "text-yellow-500", bg: "bg-yellow-500/20", label: "Low" };
    if (moisture < 70) return { color: "text-green-500", bg: "bg-green-500/20", label: "Optimal" };
    return { color: "text-blue-500", bg: "bg-blue-500/20", label: "Wet" };
  };

  const getSignalStrength = (rssi: number | undefined) => {
    if (!rssi) return { bars: 0, label: "N/A" };
    if (rssi > -50) return { bars: 4, label: "Excellent" };
    if (rssi > -60) return { bars: 3, label: "Good" };
    if (rssi > -70) return { bars: 2, label: "Fair" };
    return { bars: 1, label: "Weak" };
  };

  const formatUptime = (seconds: number | undefined) => {
    if (!seconds) return "N/A";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return (
      <div className="dashboard-card p-4">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-green-500 animate-pulse" />
          <h3 className="text-sm font-medium text-muted-foreground">Live Sensor Data</h3>
        </div>
        <div className="animate-pulse space-y-3">
          <div className="h-20 bg-muted/50 rounded-lg"></div>
          <div className="grid grid-cols-3 gap-2">
            <div className="h-16 bg-muted/50 rounded-lg"></div>
            <div className="h-16 bg-muted/50 rounded-lg"></div>
            <div className="h-16 bg-muted/50 rounded-lg"></div>
          </div>
        </div>
      </div>
    );
  }

  const moistureStatus = sensorData ? getMoistureStatus(sensorData.soil_moisture) : null;
  const signal = sensorData ? getSignalStrength(sensorData.wifi_rssi) : null;

  return (
    <div className="dashboard-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-green-500" />
          <h3 className="text-sm font-medium text-muted-foreground font-display">
            Live Sensor Data
          </h3>
          {sensorData?.is_simulated && (
            <span className="text-xs bg-yellow-500/20 text-yellow-500 px-2 py-0.5 rounded-full">
              Demo Mode
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {sensorData && !sensorData.is_simulated ? (
            <div className="flex items-center gap-1 text-xs text-green-500">
              <Wifi className="w-3 h-3" />
              <span>ESP32 Connected</span>
            </div>
          ) : (
            <div className="flex items-center gap-1 text-xs text-yellow-500">
              <WifiOff className="w-3 h-3" />
              <span>No Device</span>
            </div>
          )}
          <button
            onClick={fetchSensorData}
            className="p-1.5 rounded-lg hover:bg-muted transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      </div>

      {sensorData && (
        <>
          {/* Main Moisture Display */}
          <div className={`${moistureStatus?.bg} rounded-xl p-4 mb-4 border border-border`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Soil Moisture</p>
                <div className="flex items-baseline gap-2">
                  <span className={`text-4xl font-bold ${moistureStatus?.color}`}>
                    {sensorData.soil_moisture.toFixed(1)}
                  </span>
                  <span className="text-xl text-muted-foreground">%</span>
                </div>
                <p className={`text-sm ${moistureStatus?.color} mt-1`}>
                  {moistureStatus?.label}
                </p>
              </div>
              <div className="text-right">
                <Droplets className={`w-12 h-12 ${moistureStatus?.color} opacity-50`} />
              </div>
            </div>
            
            {/* Moisture Bar */}
            <div className="mt-3 h-2 bg-background/50 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all duration-500 ${
                  sensorData.soil_moisture < 30 ? "bg-red-500" :
                  sensorData.soil_moisture < 50 ? "bg-yellow-500" :
                  sensorData.soil_moisture < 70 ? "bg-green-500" : "bg-blue-500"
                }`}
                style={{ width: `${sensorData.soil_moisture}%` }}
              />
            </div>
          </div>

          {/* Temperature & Humidity */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-muted/50 rounded-lg p-3 text-center">
              <Thermometer className="w-5 h-5 text-orange-500 mx-auto mb-1" />
              <p className="text-2xl font-bold">{sensorData.temperature.toFixed(1)}°C</p>
              <p className="text-xs text-muted-foreground">Temperature</p>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 text-center">
              <Wind className="w-5 h-5 text-blue-500 mx-auto mb-1" />
              <p className="text-2xl font-bold">{sensorData.humidity.toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground">Humidity</p>
            </div>
          </div>

          {/* Pump Status & Control */}
          <div className="bg-muted/30 rounded-lg p-3 mb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${sensorData.pump_running ? "bg-green-500 animate-pulse" : "bg-gray-500"}`} />
                <span className="text-sm font-medium">
                  Pump: {sensorData.pump_running ? "Running" : "Off"}
                </span>
                {sensorData.pump_running && sensorData.pump_runtime > 0 && (
                  <span className="text-xs text-muted-foreground">
                    ({sensorData.pump_runtime}s)
                  </span>
                )}
              </div>
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                sensorData.auto_mode 
                  ? "bg-blue-500/20 text-blue-500" 
                  : "bg-orange-500/20 text-orange-500"
              }`}>
                {sensorData.auto_mode ? "Auto" : "Manual"}
              </span>
            </div>
            
            {/* Pump Control Buttons */}
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => sendPumpCommand("ON")}
                disabled={sensorData.is_simulated}
                className="flex items-center justify-center gap-1 py-2 px-3 rounded-lg bg-green-500/20 text-green-500 hover:bg-green-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                <Power className="w-3 h-3" />
                ON
              </button>
              <button
                onClick={() => sendPumpCommand("OFF")}
                disabled={sensorData.is_simulated}
                className="flex items-center justify-center gap-1 py-2 px-3 rounded-lg bg-red-500/20 text-red-500 hover:bg-red-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                <Power className="w-3 h-3" />
                OFF
              </button>
              <button
                onClick={() => sendPumpCommand("AUTO")}
                disabled={sensorData.is_simulated}
                className="flex items-center justify-center gap-1 py-2 px-3 rounded-lg bg-blue-500/20 text-blue-500 hover:bg-blue-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                <Cpu className="w-3 h-3" />
                AUTO
              </button>
            </div>
          </div>

          {/* Device Info */}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-3">
              {signal && !sensorData.is_simulated && (
                <div className="flex items-center gap-1">
                  <Signal className="w-3 h-3" />
                  <span>{signal.label}</span>
                </div>
              )}
              <span>Uptime: {formatUptime(sensorData.uptime)}</span>
            </div>
            <span>
              Updated: {new Date(sensorData.last_update).toLocaleTimeString()}
            </span>
          </div>
        </>
      )}

      {!sensorData && (
        <div className="text-center py-8 text-muted-foreground">
          <WifiOff className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No ESP32 device connected</p>
          <p className="text-xs mt-1">Connect your ESP32 to see live data</p>
        </div>
      )}
    </div>
  );
}
