 import { AlertTriangle, X } from "lucide-react";
 import { useState } from "react";
 
 interface AlertBannerProps {
   alerts: Array<{
     id: string;
     message: string;
     severity: "warning" | "danger";
   }>;
 }
 
 export function AlertBanner({ alerts }: AlertBannerProps) {
   const [dismissedAlerts, setDismissedAlerts] = useState<string[]>([]);
   
   const visibleAlerts = alerts.filter(alert => !dismissedAlerts.includes(alert.id));
   
   if (visibleAlerts.length === 0) return null;
 
   const dismissAlert = (id: string) => {
     setDismissedAlerts(prev => [...prev, id]);
   };
 
   return (
     <div className="space-y-2 mb-4">
       {visibleAlerts.map((alert) => (
         <div
           key={alert.id}
           className={`alert-banner rounded-xl ${
             alert.severity === "danger"
               ? "bg-destructive/10 border border-destructive/30"
               : "bg-status-warning/10 border border-status-warning/30"
           }`}
         >
           <AlertTriangle
             className={`w-5 h-5 flex-shrink-0 ${
               alert.severity === "danger" ? "text-destructive" : "text-status-warning"
             }`}
           />
           <p className="flex-1 text-sm text-foreground">{alert.message}</p>
           <button
             onClick={() => dismissAlert(alert.id)}
             className="p-1 rounded-lg hover:bg-secondary/50 transition-colors"
           >
             <X className="w-4 h-4 text-muted-foreground" />
           </button>
         </div>
       ))}
     </div>
   );
 }