 import { useState } from "react";
 import { Power, Settings, Droplets, Gauge } from "lucide-react";
 import { Switch } from "@/components/ui/switch";
 import { Button } from "@/components/ui/button";
 
 export function PumpControl() {
   const [autoMode, setAutoMode] = useState(true);
   const [pumpRunning, setPumpRunning] = useState(false);
 
   return (
    <div className="dashboard-card p-5">
       <div className="flex items-center justify-between mb-4">
         <div className="flex items-center gap-2">
           <Power className="w-5 h-5 text-muted-foreground" />
           <h3 className="text-lg font-semibold text-foreground font-display">
             Pump Control
           </h3>
         </div>
         <span className={pumpRunning ? "status-good" : "status-warning"}>
           {pumpRunning ? "Running" : "Stopped"}
         </span>
       </div>
       
       <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50 border border-border mb-4">
         <div className="flex items-center gap-2">
           <Settings className="w-4 h-4 text-muted-foreground" />
           <span className="text-sm text-foreground">Auto Mode</span>
         </div>
         <Switch
           checked={autoMode}
           onCheckedChange={setAutoMode}
         />
       </div>
       
       <Button
         className={`w-full mb-4 ${
           pumpRunning
             ? "bg-destructive hover:bg-destructive/90"
            : "bg-primary hover:bg-primary/90"
         } transition-all duration-300`}
         onClick={() => setPumpRunning(!pumpRunning)}
         disabled={autoMode}
       >
         <Power className="w-4 h-4 mr-2" />
         {pumpRunning ? "Stop Pump" : "Start Pump"}
       </Button>
       
       <div className="space-y-3">
         <div>
           <div className="flex items-center justify-between text-sm mb-1">
             <div className="flex items-center gap-2">
               <Droplets className="w-4 h-4 text-status-info" />
               <span className="text-muted-foreground">Water Level</span>
             </div>
             <span className="text-foreground font-medium">85%</span>
           </div>
           <div className="progress-bar">
             <div className="progress-bar-fill" style={{ width: "85%" }} />
           </div>
         </div>
         
         <div className="flex items-center gap-6 text-sm pt-2">
           <div className="flex items-center gap-2">
             <Gauge className="w-4 h-4 text-status-info" />
             <span className="text-muted-foreground">Pressure:</span>
             <span className="text-foreground font-medium">15 PSI</span>
           </div>
           <div className="flex items-center gap-2">
             <Droplets className="w-4 h-4 text-muted-foreground" />
             <span className="text-muted-foreground">Flow:</span>
             <span className="text-foreground font-medium">15 L/min</span>
           </div>
         </div>
       </div>
       
       {autoMode && (
        <div className="mt-4 p-3 rounded-lg bg-muted/50 border border-border flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
           <span className="text-xs text-muted-foreground">
             AI auto-control enabled
           </span>
         </div>
       )}
     </div>
   );
 }