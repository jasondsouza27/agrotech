 import { useState, useEffect } from "react";
 import { Power, Settings, Droplets, Gauge, RefreshCw } from "lucide-react";
 import { Switch } from "@/components/ui/switch";
 import { Button } from "@/components/ui/button";
 
 export function PumpControl() {
   const [autoMode, setAutoMode] = useState(true);
   const [pumpRunning, setPumpRunning] = useState(false);
   const [deviceId, setDeviceId] = useState<string | null>(null);
   const [isConnected, setIsConnected] = useState(false);

   useEffect(() => {
     const fetchPumpStatus = async () => {
       try {
         const response = await fetch("http://127.0.0.1:5000/api/esp32/sensors");
         if (response.ok) {
           const data = await response.json();
           if (data.devices && data.devices.length > 0) {
             const device = data.devices[0];
             setDeviceId(device.device_id);
             setPumpRunning(device.pump_running || false);
             setAutoMode(device.auto_mode || false);
             setIsConnected(!device.is_simulated);
           }
         }
       } catch (err) {
         console.log("ESP32 not connected");
         setIsConnected(false);
       }
     };

     fetchPumpStatus();
     const interval = setInterval(fetchPumpStatus, 250);
     return () => clearInterval(interval);
   }, []);

   const sendPumpCommand = async (command: "ON" | "OFF" | "AUTO") => {
     if (!deviceId) return;
     
     try {
       const response = await fetch("http://127.0.0.1:5000/api/esp32/pump", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           device_id: deviceId,
           command: command
         })
       });
       
       if (response.ok) {
         const result = await response.json();
         console.log("Pump command sent:", result);
         
         // Update local state immediately
         if (command === "AUTO") {
           setAutoMode(true);
         } else {
           setAutoMode(false);
           setPumpRunning(command === "ON");
         }
       }
     } catch (err) {
       console.error("Failed to send pump command:", err);
     }
   };

   const handleAutoModeToggle = (checked: boolean) => {
     if (checked) {
       sendPumpCommand("AUTO");
     } else {
       setAutoMode(false);
       // Don't send command, just disable auto mode locally
     }
   };

   const handlePumpToggle = () => {
     const command = pumpRunning ? "OFF" : "ON";
     sendPumpCommand(command);
   };
 
   return (
    <div className="dashboard-card p-5">
       <div className="flex items-center justify-between mb-4">
         <div className="flex items-center gap-2">
           <Power className="w-5 h-5 text-muted-foreground" />
           <h3 className="text-lg font-semibold text-foreground font-display">
             Pump Control
           </h3>
           {!isConnected && (
             <span className="text-xs bg-yellow-500/20 text-yellow-500 px-2 py-0.5 rounded-full">
               Demo
             </span>
           )}
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
           onCheckedChange={handleAutoModeToggle}
           disabled={!isConnected}
         />
       </div>
       
       <Button
         className={`w-full mb-4 ${
           pumpRunning
             ? "bg-destructive hover:bg-destructive/90"
            : "bg-primary hover:bg-primary/90"
         } transition-all duration-300`}
         onClick={handlePumpToggle}
         disabled={autoMode || !isConnected}
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
       
       {!isConnected && (
         <div className="mt-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20 flex items-center gap-2">
           <RefreshCw className="w-3 h-3 text-yellow-500" />
           <span className="text-xs text-yellow-500">
             Connect ESP32 for live control
           </span>
         </div>
       )}
     </div>
   );
 }