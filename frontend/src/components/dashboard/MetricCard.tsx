 import { LucideIcon } from "lucide-react";
 import { cn } from "@/lib/utils";
 
 interface MetricCardProps {
   title: string;
   value: string | number;
   unit: string;
   icon: LucideIcon;
   status: "good" | "warning" | "danger";
   change?: string;
   onClick?: () => void;
 }
 
 export function MetricCard({
   title,
   value,
   unit,
   icon: Icon,
   status,
   change,
   onClick,
 }: MetricCardProps) {
   const statusLabels = {
     good: "Good",
     warning: "Warning",
     danger: "Critical",
   };
 
   return (
     <div
       className={cn(
        "dashboard-card p-5 cursor-pointer transition-all duration-200 hover:bg-secondary/50",
         onClick && "cursor-pointer"
       )}
       onClick={onClick}
     >
       <div className="flex items-start justify-between mb-3">
         <span className="text-sm text-muted-foreground font-medium">{title}</span>
         <Icon className="w-5 h-5 text-muted-foreground" />
       </div>
       
       <div className="flex items-baseline gap-1 mb-3">
         <span className="metric-value text-foreground">{value}</span>
         <span className="text-lg text-muted-foreground">{unit}</span>
       </div>
       
       <div className="flex items-center justify-between">
         <span className={`status-${status}`}>{statusLabels[status]}</span>
         {change && (
           <span className="text-xs text-muted-foreground">{change}</span>
         )}
       </div>
     </div>
   );
 }