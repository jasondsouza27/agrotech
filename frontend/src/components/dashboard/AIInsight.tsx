 import { useState, useEffect, useCallback } from "react";
 import { Brain, Clock, Droplets, Zap, Leaf, Settings, RefreshCw, ChevronDown, Filter } from "lucide-react";
 
 interface InsightLog {
   timestamp: string;
   action: string;
   reason: string;
   category?: string;
   sensors?: {
     soil_moisture?: number;
     temperature?: number;
     humidity?: number;
   };
 }

const CATEGORY_CONFIG: Record<string, { icon: typeof Brain; color: string; label: string }> = {
  pump: { icon: Droplets, color: "text-blue-400", label: "Pump" },
  ml: { icon: Zap, color: "text-yellow-400", label: "ML Decision" },
  system: { icon: Settings, color: "text-purple-400", label: "System" },
  scan: { icon: Leaf, color: "text-green-400", label: "Leaf Scan" },
  sensor: { icon: Brain, color: "text-cyan-400", label: "Sensor" },
};
 
 export function AIInsight() {
   const [events, setEvents] = useState<InsightLog[]>([]);
   const [loading, setLoading] = useState(true);
   const [expanded, setExpanded] = useState(false);
   const [filterCategory, setFilterCategory] = useState<string | null>(null);

   const fetchEvents = useCallback(async () => {
     try {
       const params = new URLSearchParams({ limit: "50" });
       if (filterCategory) params.set("category", filterCategory);
       const res = await fetch(`/api/esp32/events?${params}`);
       if (res.ok) {
         const data = await res.json();
         setEvents(data.events || []);
       }
     } catch (err) {
       console.error("Failed to fetch events:", err);
     } finally {
       setLoading(false);
     }
   }, [filterCategory]);

   useEffect(() => {
     fetchEvents();
     const interval = setInterval(fetchEvents, 5000); // Poll every 5s
     return () => clearInterval(interval);
   }, [fetchEvents]);

   const formatTimestamp = (ts: string) => {
     try {
       const date = new Date(ts);
       const now = new Date();
       const diffMs = now.getTime() - date.getTime();
       const diffMin = Math.floor(diffMs / 60000);
       const diffHr = Math.floor(diffMin / 60);

       if (diffMin < 1) return "Just now";
       if (diffMin < 60) return `${diffMin}m ago`;
       if (diffHr < 24) return `${diffHr}h ago`;
       return date.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
     } catch {
       return ts;
     }
   };

   const getTimeDisplay = (ts: string) => {
     try {
       return new Date(ts).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false });
     } catch {
       return ts;
     }
   };

   const displayedEvents = expanded ? events : events.slice(0, 5);
   const hasMore = events.length > 5;

   return (
    <div className="dashboard-card p-4 mt-6">
       <div className="flex items-center gap-2 mb-4">
         <Brain className="w-5 h-5 text-muted-foreground" />
         <h3 className="text-sm font-medium text-muted-foreground font-display">
           AI System Insight Log
         </h3>
         <span className="text-xs text-muted-foreground/60 ml-auto flex items-center gap-2">
           {loading && <RefreshCw className="w-3 h-3 animate-spin" />}
           Transparency & Audit Trail
         </span>
       </div>

       {/* Category filter chips */}
       <div className="flex items-center gap-1.5 mb-3 flex-wrap">
         <Filter className="w-3 h-3 text-muted-foreground" />
         <button
           onClick={() => setFilterCategory(null)}
           className={`px-2 py-0.5 rounded-full text-xs transition-colors ${
             !filterCategory ? "bg-accent/20 text-accent border border-accent/40" : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
           }`}
         >
           All
         </button>
         {Object.entries(CATEGORY_CONFIG).map(([key, config]) => (
           <button
             key={key}
             onClick={() => setFilterCategory(filterCategory === key ? null : key)}
             className={`px-2 py-0.5 rounded-full text-xs transition-colors ${
               filterCategory === key ? "bg-accent/20 text-accent border border-accent/40" : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
             }`}
           >
             {config.label}
           </button>
         ))}
       </div>
       
       <div className="space-y-2">
         {displayedEvents.length === 0 ? (
           <div className="text-center py-6 text-muted-foreground/60 text-sm">
             {loading ? "Loading events..." : "No events yet. Events will appear as the system operates."}
           </div>
         ) : (
           displayedEvents.map((event, index) => {
             const catConfig = CATEGORY_CONFIG[event.category || "system"] || CATEGORY_CONFIG.system;
             const CatIcon = catConfig.icon;
             return (
               <div
                 key={`${event.timestamp}-${index}`}
                 className="flex gap-3 p-3 rounded-lg bg-muted/50 border border-border hover:bg-muted/70 transition-colors"
               >
                 <div className="flex flex-col items-center gap-1 flex-shrink-0 pt-0.5">
                   <CatIcon className={`w-4 h-4 ${catConfig.color}`} />
                   <div className="flex items-center gap-1 text-xs text-muted-foreground">
                     <Clock className="w-3 h-3" />
                     {getTimeDisplay(event.timestamp)}
                   </div>
                   <span className="text-[10px] text-muted-foreground/50">
                     {formatTimestamp(event.timestamp)}
                   </span>
                 </div>
                 <div className="flex-1 min-w-0">
                   <div className="flex items-center gap-2 mb-0.5">
                     <p className="text-sm font-medium text-foreground/80">
                       {event.action}
                     </p>
                     <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${catConfig.color} bg-current/10 border border-current/20`}
                           style={{ backgroundColor: `color-mix(in srgb, currentColor 10%, transparent)` }}>
                       {catConfig.label}
                     </span>
                   </div>
                   <p className="text-xs text-muted-foreground leading-relaxed">
                     {event.reason}
                   </p>
                   {event.sensors && (
                     <div className="flex items-center gap-3 mt-1.5 text-[11px] text-muted-foreground/70">
                       {event.sensors.soil_moisture != null && (
                         <span>💧 {event.sensors.soil_moisture}%</span>
                       )}
                       {event.sensors.temperature != null && (
                         <span>🌡️ {event.sensors.temperature}°C</span>
                       )}
                       {event.sensors.humidity != null && (
                         <span>💨 {event.sensors.humidity}%</span>
                       )}
                     </div>
                   )}
                 </div>
               </div>
             );
           })
         )}
       </div>

       {/* Show more / less */}
       {hasMore && (
         <button
           onClick={() => setExpanded(!expanded)}
           className="w-full mt-2 py-1.5 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center gap-1 transition-colors"
         >
           <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? "rotate-180" : ""}`} />
           {expanded ? "Show less" : `Show ${events.length - 5} more events`}
         </button>
       )}
       
       <p className="text-xs text-muted-foreground/60 mt-3 text-center">
         All AI decisions are logged for transparency and review
       </p>
     </div>
   );
 }