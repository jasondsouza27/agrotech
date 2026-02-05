import { useState } from "react";
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
import { signOut } from "@/lib/supabase";

const alerts = [
  {
    id: "1",
    message: "Disease risk elevated: Fungal infection probability 67% in Sector B due to high humidity.",
    severity: "warning" as const,
  },
];

const Index = () => {
  const navigate = useNavigate();

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
            value={28}
            unit="°C"
            icon={Thermometer}
            status="good"
            change="+1°C from yesterday"
          />
          <MetricCard
            title="Wind Speed"
            value={12}
            unit="km/h"
            icon={Wind}
            status="good"
            change="-3% from yesterday"
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
            <WeatherWidget />
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
