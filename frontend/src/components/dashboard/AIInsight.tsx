 import { Brain, Clock } from "lucide-react";
 
 interface InsightLog {
   timestamp: string;
   action: string;
   reason: string;
 }
 
 const recentInsights: InsightLog[] = [
   {
     timestamp: "14:32",
     action: "Paused irrigation",
     reason: "High noon heat stress detected. Resuming at 16:00 to prevent water evaporation.",
   },
   {
     timestamp: "11:45",
     action: "Extended watering cycle",
     reason: "Soil moisture below optimal threshold. Added 15 minutes to morning irrigation.",
   },
   {
     timestamp: "08:00",
     action: "Started morning cycle",
     reason: "Optimal conditions: low temperature, calm winds, ideal absorption rate.",
   },
 ];
 
 export function AIInsight() {
   return (
    <div className="dashboard-card p-4 mt-6">
       <div className="flex items-center gap-2 mb-4">
         <Brain className="w-5 h-5 text-muted-foreground" />
         <h3 className="text-sm font-medium text-muted-foreground font-display">
           AI System Insight Log
         </h3>
         <span className="text-xs text-muted-foreground/60 ml-auto">
           Transparency & Audit Trail
         </span>
       </div>
       
       <div className="space-y-3">
         {recentInsights.map((insight, index) => (
           <div
             key={index}
            className="flex gap-3 p-3 rounded-lg bg-muted/50 border border-border"
           >
             <div className="flex items-center gap-1 text-xs text-muted-foreground flex-shrink-0">
               <Clock className="w-3 h-3" />
               {insight.timestamp}
             </div>
             <div className="flex-1 min-w-0">
               <p className="text-sm font-medium text-foreground/80 mb-0.5">
                 {insight.action}
               </p>
               <p className="text-xs text-muted-foreground leading-relaxed">
                 {insight.reason}
               </p>
             </div>
           </div>
         ))}
       </div>
       
       <p className="text-xs text-muted-foreground/60 mt-3 text-center">
         All AI decisions are logged for transparency and review
       </p>
     </div>
   );
 }