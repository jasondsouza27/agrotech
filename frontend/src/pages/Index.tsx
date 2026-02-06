import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Clock, Droplets, Thermometer, Wind, Beaker, LogOut, Wifi } from "lucide-react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SensorChart } from "@/components/dashboard/SensorChart";
import { WeatherWidget } from "@/components/dashboard/WeatherWidget";
import { LeafDoctor } from "@/components/dashboard/LeafDoctor";
import { SustainabilityMetrics } from "@/components/dashboard/SustainabilityMetrics";
import { AlertBanner } from "@/components/dashboard/AlertBanner";
import { AIInsight } from "@/components/dashboard/AIInsight";
import { RegenChat } from "@/components/dashboard/RegenChat";
import { MandiConnect } from "@/components/dashboard/MandiConnect";
import { CropRecommendation } from "@/components/dashboard/CropRecommendation";
import { LiveSensorData } from "@/components/dashboard/LiveSensorData";
import { Button } from "@/components/ui/button";
import { signOut, supabase } from "@/lib/supabase";

interface LiveMetrics {
  soil_moisture: number;
  temperature: number;
  humidity: number;
  is_live: boolean;
}

const Index = () => {
  const navigate = useNavigate();
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics>({
    soil_moisture: 52,
    temperature: 28,
    humidity: 65,
    is_live: false
  });
  const [alerts, setAlerts] = useState([
    {
      id: "1",
      message: "Disease risk elevated: Fungal infection probability 67% in Sector B due to high humidity.",
      severity: "warning" as const,
    },
  ]);
  const [windSpeed, setWindSpeed] = useState(12);
  const [windChange, setWindChange] = useState("");
  const [userDistrict, setUserDistrict] = useState<string>("");

  // Fetch user's district from Supabase
  useEffect(() => {
    const fetchUserDistrict = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          // First try from user metadata (set during signup)
          const metaDistrict = user.user_metadata?.district;
          if (metaDistrict) {
            setUserDistrict(metaDistrict);
            return;
          }
          // Fallback: fetch from farmers table
          const { data: profile } = await supabase
            .from("farmers")
            .select("district")
            .eq("user_id", user.id)
            .single();
          if (profile?.district) {
            setUserDistrict(profile.district);
          }
        }
      } catch {
        // Use default location if Supabase is unavailable
      }
    };
    fetchUserDistrict();
  }, []);

  // Fetch live sensor data from ESP32
  useEffect(() => {
    const fetchLiveData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/api/esp32/sensors");
        if (response.ok) {
          const data = await response.json();
          if (data.devices && data.devices.length > 0) {
            const device = data.devices[0];
            setLiveMetrics({
              soil_moisture: device.soil_moisture || 52,
              temperature: device.temperature || 28,
              humidity: device.humidity || 65,
              is_live: !device.is_simulated
            });

            // Generate alerts based on live data
            const newAlerts = [];
            if (device.soil_moisture < 30) {
              newAlerts.push({
                id: "moisture",
                message: `⚠️ Soil moisture critically low at ${device.soil_moisture.toFixed(1)}%! Irrigation recommended.`,
                severity: "critical" as const
              });
            }
            if (device.temperature > 35) {
              newAlerts.push({
                id: "temp",
                message: `🌡️ High temperature alert: ${device.temperature.toFixed(1)}°C. Consider shade protection.`,
                severity: "warning" as const
              });
            }
            if (device.humidity > 85) {
              newAlerts.push({
                id: "humidity",
                message: `💧 High humidity (${device.humidity.toFixed(1)}%) increases fungal disease risk.`,
                severity: "warning" as const
              });
            }
            if (newAlerts.length > 0) {
              setAlerts(newAlerts);
            }
          }
        }
      } catch (err) {
        console.log("Using default metrics - ESP32 not connected");
      }
    };

    fetchLiveData();
    const interval = setInterval(fetchLiveData, 250);
    return () => clearInterval(interval);
  }, []);

  // Fetch weather using user's district (for wind speed)
  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const params = userDistrict ? `?district=${encodeURIComponent(userDistrict)}` : "";
        const res = await fetch(`http://127.0.0.1:5000/api/weather${params}`);
        const data = await res.json();
        if (data.success) {
          const c = data.current;
          setWindSpeed(c.wind_speed);
          setWindChange(c.weather_description);
        }
      } catch {
        // Keep default values on error
      }
    };
    fetchWeather();
    const interval = setInterval(fetchWeather, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [userDistrict]);

  const handleLogout = async () => {
    await signOut();
    navigate("/login");
  };

  const getMoistureStatus = (m: number): "warning" | "good" | "danger" => 
    m < 30 ? "danger" : m < 50 ? "warning" : "good";
  const getTempStatus = (t: number): "warning" | "good" | "danger" => 
    t > 38 ? "danger" : t > 35 ? "warning" : "good";
  const getHumidityStatus = (h: number): "warning" | "good" | "danger" => 
    h > 90 ? "warning" : h < 30 ? "warning" : "good";

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🌿</span>
            <div>
              <h1 className="text-2xl font-bold text-gradient-emerald font-display">
                AgroSmart
              </h1>
              <p className="text-sm text-muted-foreground">
                Green Growth Edition • Intelligent Farm Management
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" className="gap-2">
              <Clock className="w-4 h-4" />
              Historical Logs
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={handleLogout}
              className="gap-2 text-muted-foreground hover:text-destructive"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* Alert Banner */}
        <AlertBanner alerts={alerts} />

        {/* Live Data Indicator */}
        {liveMetrics.is_live && (
          <div className="flex items-center gap-2 mb-4 text-green-500 text-sm">
            <Wifi className="w-4 h-4" />
            <span>Live data from ESP32</span>
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
          </div>
        )}

        {/* Core Metrics - Top Priority (LIVE DATA) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Soil Moisture"
            value={Number(liveMetrics.soil_moisture.toFixed(1))}
            unit="%"
            icon={Droplets}
            status={getMoistureStatus(liveMetrics.soil_moisture)}
            change={liveMetrics.is_live ? "🔴 Live" : "Demo data"}
          />
          <MetricCard
            title="Temperature"
            value={Number(liveMetrics.temperature.toFixed(1))}
            unit="°C"
            icon={Thermometer}
            status={getTempStatus(liveMetrics.temperature)}
            change={liveMetrics.is_live ? "🔴 Live" : "Demo data"}
          />
          <MetricCard
            title="Humidity"
            value={Number(liveMetrics.humidity.toFixed(1))}
            unit="%"
            icon={Wind}
            status={getHumidityStatus(liveMetrics.humidity)}
            change={liveMetrics.is_live ? "🔴 Live" : "Demo data"}
          />
          <MetricCard
            title="Soil pH"
            value={6.8}
            unit="pH"
            icon={Beaker}
            status="good"
            change="Stable"
          />
        </div>

        {/* Bento Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Live Sensor Data from ESP32 */}
          <div>
            <LiveSensorData />
          </div>

          {/* Sensor Chart - Spans 2 columns */}
          <div className="lg:col-span-2">
            <SensorChart />
          </div>

          {/* Weather Widget */}
          <div>
            <WeatherWidget district={userDistrict} />
          </div>

          {/* Sustainability Metrics */}
          <div>
            <SustainabilityMetrics />
          </div>

          {/* Crop Recommendation - ML Powered */}
          <div>
            <CropRecommendation />
          </div>

          {/* Mandi Connect - Market Prices */}
          <div className="lg:col-span-2 lg:row-span-2">
            <MandiConnect />
          </div>

          {/* Leaf Doctor */}
          <div>
            <LeafDoctor />
          </div>
        </div>

        {/* AI Insight - Bottom, Low Priority */}
        <AIInsight />
      </main>

      {/* Floating Chat */}
      <RegenChat />
    </div>
  );
};

export default Index;
