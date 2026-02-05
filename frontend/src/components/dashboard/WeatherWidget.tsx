 import { Sun, Cloud, Droplets, Wind } from "lucide-react";
 
 export function WeatherWidget() {
   return (
    <div className="dashboard-card p-5">
       <div className="flex items-center gap-2 mb-4">
         <Sun className="w-5 h-5 text-status-warning" />
         <h3 className="text-lg font-semibold text-foreground font-display">
           Current Weather
         </h3>
       </div>
       
       <div className="flex items-center justify-between mb-4">
         <div className="p-3 rounded-xl bg-status-warning/10">
           <Sun className="w-10 h-10 text-status-warning" />
         </div>
         <div className="text-right">
           <span className="metric-value text-foreground">28°C</span>
           <p className="text-sm text-muted-foreground">Sunny</p>
         </div>
       </div>
       
       <div className="flex items-center gap-6 mb-4 text-sm">
         <div className="flex items-center gap-2">
           <Droplets className="w-4 h-4 text-status-info" />
           <span className="text-muted-foreground">Humidity:</span>
           <span className="text-foreground font-medium">65%</span>
         </div>
         <div className="flex items-center gap-2">
           <Wind className="w-4 h-4 text-muted-foreground" />
           <span className="text-muted-foreground">Wind:</span>
           <span className="text-foreground font-medium">12 km/h</span>
         </div>
       </div>
       
       <div className="p-3 rounded-lg bg-secondary/50 border border-border">
         <p className="text-xs text-muted-foreground mb-1">24h Forecast</p>
         <p className="text-sm text-foreground">
           Sunny with occasional clouds. Perfect conditions for irrigation.
         </p>
       </div>
     </div>
   );
 }