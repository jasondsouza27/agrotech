import { useState, useEffect, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Brain,
  Clock,
  Droplets,
  Zap,
  Leaf,
  Settings,
  RefreshCw,
  Filter,
  Download,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface EventLog {
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

const CATEGORY_CONFIG: Record<
  string,
  { icon: typeof Brain; color: string; label: string }
> = {
  pump: { icon: Droplets, color: "text-blue-400", label: "Pump" },
  ml: { icon: Zap, color: "text-yellow-400", label: "ML Decision" },
  system: { icon: Settings, color: "text-purple-400", label: "System" },
  scan: { icon: Leaf, color: "text-green-400", label: "Leaf Scan" },
  sensor: { icon: Brain, color: "text-cyan-400", label: "Sensor" },
};

interface HistoricalLogsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function HistoricalLogsModal({
  open,
  onOpenChange,
}: HistoricalLogsModalProps) {
  const [events, setEvents] = useState<EventLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterCategory, setFilterCategory] = useState<string | null>(null);

  const fetchEvents = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: "200" });
      if (filterCategory) params.set("category", filterCategory);
      const res = await fetch(`/api/esp32/events?${params}`);
      if (res.ok) {
        const data = await res.json();
        setEvents(data.events || []);
      }
    } catch (err) {
      console.error("Failed to fetch historical events:", err);
    } finally {
      setLoading(false);
    }
  }, [filterCategory]);

  useEffect(() => {
    if (open) {
      setLoading(true);
      fetchEvents();
      const interval = setInterval(fetchEvents, 10000);
      return () => clearInterval(interval);
    }
  }, [open, fetchEvents]);

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
      return date.toLocaleDateString("en-IN", {
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return ts;
    }
  };

  const getFullTimestamp = (ts: string) => {
    try {
      return new Date(ts).toLocaleString("en-IN", {
        day: "numeric",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
      });
    } catch {
      return ts;
    }
  };

  const handleExport = () => {
    const lines = events.map(
      (e) =>
        `[${e.timestamp}] [${e.category || "system"}] ${e.action} — ${e.reason}${
          e.sensors
            ? ` | Moisture: ${e.sensors.soil_moisture ?? "-"}%, Temp: ${e.sensors.temperature ?? "-"}°C, Humidity: ${e.sensors.humidity ?? "-"}%`
            : ""
        }`
    );
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `agrosmart-logs-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-accent" />
            Historical System Logs
          </DialogTitle>
          <DialogDescription>
            Complete audit trail of all AI decisions, pump actions, and system
            events.
          </DialogDescription>
        </DialogHeader>

        {/* Toolbar: filter chips + export */}
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 flex-wrap">
            <Filter className="w-3 h-3 text-muted-foreground" />
            <button
              onClick={() => setFilterCategory(null)}
              className={`px-2 py-0.5 rounded-full text-xs transition-colors ${
                !filterCategory
                  ? "bg-accent/20 text-accent border border-accent/40"
                  : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
              }`}
            >
              All
            </button>
            {Object.entries(CATEGORY_CONFIG).map(([key, config]) => (
              <button
                key={key}
                onClick={() =>
                  setFilterCategory(filterCategory === key ? null : key)
                }
                className={`px-2 py-0.5 rounded-full text-xs transition-colors ${
                  filterCategory === key
                    ? "bg-accent/20 text-accent border border-accent/40"
                    : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
                }`}
              >
                {config.label}
              </button>
            ))}
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={handleExport}
            disabled={events.length === 0}
          >
            <Download className="w-3 h-3" />
            Export
          </Button>
        </div>

        {/* Scrollable event list */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 min-h-0">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-muted-foreground gap-2">
              <RefreshCw className="w-4 h-4 animate-spin" />
              Loading events…
            </div>
          ) : events.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground/60 text-sm">
              No events recorded yet. Events will appear as the system operates.
            </div>
          ) : (
            events.map((event, index) => {
              const catConfig =
                CATEGORY_CONFIG[event.category || "system"] ||
                CATEGORY_CONFIG.system;
              const CatIcon = catConfig.icon;
              return (
                <div
                  key={`${event.timestamp}-${index}`}
                  className="flex gap-3 p-3 rounded-lg bg-muted/50 border border-border hover:bg-muted/70 transition-colors"
                >
                  <div className="flex flex-col items-center gap-1 flex-shrink-0 pt-0.5">
                    <CatIcon className={`w-4 h-4 ${catConfig.color}`} />
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded-full ${catConfig.color}`}
                      style={{
                        backgroundColor: `color-mix(in srgb, currentColor 10%, transparent)`,
                      }}
                    >
                      {catConfig.label}
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-0.5">
                      <p className="text-sm font-medium text-foreground/80">
                        {event.action}
                      </p>
                      <span
                        className="text-[11px] text-muted-foreground/60 flex-shrink-0"
                        title={getFullTimestamp(event.timestamp)}
                      >
                        {formatTimestamp(event.timestamp)}
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

        {/* Footer */}
        <div className="pt-2 border-t border-border flex items-center justify-between text-xs text-muted-foreground/60">
          <span>
            {events.length} event{events.length !== 1 ? "s" : ""} loaded
          </span>
          <span>Auto-refreshes every 10s</span>
        </div>
      </DialogContent>
    </Dialog>
  );
}
