 import { useState } from "react";
 import { Clock, Droplets, Thermometer, Wind, Beaker } from "lucide-react";
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
 import { Button } from "@/components/ui/button";
 
 const alerts = [
   {
     id: "1",
     message: "Disease risk elevated: Fungal infection probability 67% in Sector B due to high humidity.",
     severity: "warning" as const,
   },
 ];
 
 const Index = () => {
   return (
     <div className="min-h-screen bg-background">
       {/* Header */}
       <header className="border-b border-border px-6 py-4">
         <div className="max-w-7xl mx-auto flex items-center justify-between">
           <div>
             <h1 className="text-2xl font-bold text-gradient-emerald font-display">
               AgroSmart Dashboard
             </h1>
             <p className="text-sm text-muted-foreground">
               Intelligent Irrigation System • Monitor, Control & Optimize
             </p>
           </div>
           <Button variant="outline" className="gap-2">
             <Clock className="w-4 h-4" />
             Historical Logs
           </Button>
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
             change="+1.5°C from yesterday"
           />
           <MetricCard
             title="Air Humidity"
             value={66}
             unit="%"
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
