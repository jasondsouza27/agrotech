import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Clock, Droplets, Thermometer, Wind, Beaker, LogOut } from "lucide-react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SensorChart } from "@/components/dashboard/SensorChart";
import { WeatherWidget } from "@/components/dashboard/WeatherWidget";
import { PumpControl } from "@/components/dashboard/PumpControl";
import { LeafDoctor } from "@/components/dashboard/LeafDoctor";
import { SustainabilityMetrics } from "@/components/dashboard/SustainabilityMetrics";
import { AlertBanner } from "@/components/dashboard/AlertBanner";
import { AIInsight } from "@/components/dashboard/AIInsight";
import { RegenChat } from "@/components/dashboard/RegenChat";
import { MandiConnect } from "@/components/dashboard/MandiConnect";
import { CropRecommendation } from "@/components/dashboard/CropRecommendation";
import { Button } from "@/components/ui/button";
import { signOut, supabase } from "@/lib/supabase";

const alerts = [
  {
    id: "1",
    message: "Disease risk elevated: Fungal infection probability 67% in Sector B due to high humidity.",
    severity: "warning" as const,
  },
];

const Index = () => {
  const navigate = useNavigate();
  const [temperature, setTemperature] = useState(28);
  const [humidity, setHumidity] = useState(65);
  const [windSpeed, setWindSpeed] = useState(12);
  const [tempChange, setTempChange] = useState("");
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

  // Fetch weather using user's district
  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const params = userDistrict ? `?district=${encodeURIComponent(userDistrict)}` : "";
        const res = await fetch(`http://127.0.0.1:5000/api/weather${params}`);
        const data = await res.json();
        if (data.success) {
          const c = data.current;
          setTemperature(Math.round(c.temperature));
          setHumidity(c.humidity);
          setWindSpeed(c.wind_speed);
          const feelsLikeDiff = Math.round(c.feels_like - c.temperature);
          setTempChange(feelsLikeDiff >= 0 ? `Feels like +${feelsLikeDiff}°C` : `Feels like ${feelsLikeDiff}°C`);
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

        {/* Core Metrics - Top Priority */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Soil Moisture"
            value={52}
            unit="%"
            icon={Droplets}
            status="warning"
            change="+2% from yesterday"
          />
          <MetricCard
            title="Temperature"
            value={temperature}
            unit="°C"
            icon={Thermometer}
            status={temperature > 35 ? "danger" : temperature > 30 ? "warning" : "good"}
            change={tempChange || `Humidity: ${humidity}%`}
          />
          <MetricCard
            title="Wind Speed"
            value={windSpeed}
            unit="km/h"
            icon={Wind}
            status={windSpeed > 40 ? "danger" : windSpeed > 25 ? "warning" : "good"}
            change={windChange || "Live data"}
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

          {/* Mandi Connect - Market Prices */}
          <div className="lg:col-span-2 lg:row-span-2">
            <MandiConnect />
          </div>

          {/* Pump Control */}
          <div>
            <PumpControl />
          </div>

          {/* Crop Recommendation - ML Powered */}
          <div>
            <CropRecommendation />
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
