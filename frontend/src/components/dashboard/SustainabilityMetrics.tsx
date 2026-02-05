 import { Droplets, Leaf, TrendingUp } from "lucide-react";
 
 export function SustainabilityMetrics() {
   // Calculate water saved: Hours pump paused × 20 liters
   const hoursPaused = 142;
   const waterSaved = hoursPaused * 20;
 
   return (
    <div className="dashboard-card p-5">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-2 rounded-lg bg-accent/10">
          <Leaf className="w-4 h-4 text-accent" />
         </div>
        <h3 className="text-lg font-semibold text-foreground font-display">
          Sustainability Impact
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-4 rounded-xl bg-secondary/50 border border-border">
          <div className="flex items-center gap-2 mb-2">
            <Droplets className="w-4 h-4 text-status-info" />
            <span className="text-xs text-muted-foreground">Water Saved</span>
           </div>
          <p className="text-2xl font-bold text-accent font-display">
            {waterSaved.toLocaleString()} L
          </p>
          <p className="text-xs text-muted-foreground mt-1">This month</p>
         </div>

        <div className="p-4 rounded-xl bg-secondary/50 border border-border">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-status-good" />
            <span className="text-xs text-muted-foreground">Efficiency</span>
           </div>
          <p className="text-2xl font-bold text-status-good font-display">
            +23%
          </p>
          <p className="text-xs text-muted-foreground mt-1">vs. last month</p>
        </div>
      </div>

      <div className="p-3 rounded-lg bg-muted/50 border border-border">
        <div className="flex items-start gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-accent mt-1.5" />
          <p className="text-xs text-muted-foreground">
            AI-optimized irrigation has reduced water consumption by <span className="text-foreground font-medium">142 hours</span> of unnecessary pump operation.
          </p>
         </div>
       </div>
     </div>
   );
 }