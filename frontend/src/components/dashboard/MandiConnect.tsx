 import { TrendingUp, MapPin, IndianRupee, ArrowUpRight } from "lucide-react";
 
 interface MandiPrice {
   mandi: string;
   location: string;
   price: number;
   change: number;
 }
 
 const mandiPrices: MandiPrice[] = [
   { mandi: "Nashik Mandi", location: "Maharashtra", price: 40, change: 5.2 },
   { mandi: "Azadpur Mandi", location: "Delhi", price: 38, change: 3.1 },
   { mandi: "Vashi APMC", location: "Mumbai", price: 32, change: -2.4 },
   { mandi: "Kolar Mandi", location: "Karnataka", price: 35, change: 1.8 },
 ];
 
 const localTraderPrice = 15;
 const recommendedCrop = "Organic Tomatoes";
 
 export function MandiConnect() {
   const bestPrice = Math.max(...mandiPrices.map((m) => m.price));
   const potentialGain = bestPrice - localTraderPrice;
   const percentageGain = ((potentialGain / localTraderPrice) * 100).toFixed(0);
 
   return (
     <div className="dashboard-card p-5">
       <div className="flex items-center justify-between mb-4">
         <div className="flex items-center gap-2">
           <div className="p-2 rounded-lg bg-accent/10">
             <TrendingUp className="w-5 h-5 text-accent" />
           </div>
           <div>
             <h3 className="font-semibold text-foreground">Mandi Connect</h3>
             <p className="text-xs text-muted-foreground">Live Market Prices</p>
           </div>
         </div>
         <span className="text-xs text-muted-foreground">Updated 10 min ago</span>
       </div>
 
       {/* Recommended Crop */}
       <div className="mb-4 p-3 rounded-lg bg-muted/50 border border-border">
         <p className="text-sm text-muted-foreground mb-1">Recommended Crop</p>
         <p className="font-semibold text-foreground">{recommendedCrop}</p>
       </div>
 
       {/* Mandi Prices List */}
       <div className="space-y-2 mb-4">
         {mandiPrices.map((mandi) => (
           <div
             key={mandi.mandi}
             className="flex items-center justify-between p-3 rounded-lg bg-secondary/50 border border-border hover:bg-secondary/80 transition-colors"
           >
             <div className="flex items-center gap-2">
               <MapPin className="w-4 h-4 text-muted-foreground" />
               <div>
                 <p className="text-sm font-medium text-foreground">{mandi.mandi}</p>
                 <p className="text-xs text-muted-foreground">{mandi.location}</p>
               </div>
             </div>
             <div className="text-right">
               <div className="flex items-center gap-1">
                 <IndianRupee className="w-3 h-3 text-foreground" />
                 <span className="font-semibold text-foreground">{mandi.price}/kg</span>
               </div>
               <span
                 className={`text-xs ${mandi.change >= 0 ? "text-status-good" : "text-status-danger"}`}
               >
                 {mandi.change >= 0 ? "+" : ""}
                 {mandi.change}%
               </span>
             </div>
           </div>
         ))}
       </div>
 
       {/* Local Trader Comparison */}
       <div className="p-3 rounded-lg border border-dashed border-status-warning/50 bg-status-warning/5 mb-4">
         <div className="flex items-center justify-between">
           <div>
             <p className="text-sm text-muted-foreground">Local Trader (estimated)</p>
             <div className="flex items-center gap-1">
               <IndianRupee className="w-3 h-3 text-status-warning" />
               <span className="font-semibold text-status-warning">{localTraderPrice}/kg</span>
             </div>
           </div>
           <div className="text-right">
             <p className="text-xs text-muted-foreground">Potential Loss</p>
             <p className="text-sm font-medium text-status-danger">-₹{potentialGain}/kg</p>
           </div>
         </div>
       </div>
 
       {/* Income Gain Highlight */}
       <div className="p-4 rounded-lg bg-accent/10 border border-accent/30">
         <div className="flex items-center justify-between">
           <div>
             <p className="text-sm text-muted-foreground mb-1">Sell at Best Mandi Price</p>
             <p className="text-lg font-bold text-accent">
               +{percentageGain}% Higher Income
             </p>
           </div>
           <div className="flex items-center gap-1 text-accent">
             <ArrowUpRight className="w-5 h-5" />
             <span className="font-semibold">₹{potentialGain}/kg more</span>
           </div>
         </div>
       </div>
 
       <p className="text-xs text-muted-foreground mt-3 text-center">
         Data source: data.gov.in • Open Government Data Platform
       </p>
     </div>
   );
 }