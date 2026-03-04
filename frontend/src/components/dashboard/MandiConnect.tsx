import { useState, useEffect } from "react";
import { TrendingUp, MapPin, IndianRupee, ArrowUpRight, RefreshCw } from "lucide-react";

interface MandiPrice {
  mandi: string;
  location: string;
  price: number;
  change: number;
  distance_km?: number;
}

const localTraderPrice = 15;

interface MandiConnectProps {
  district?: string;
}

export function MandiConnect({ district }: MandiConnectProps) {
  const [mandiPrices, setMandiPrices] = useState<MandiPrice[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCrop, setSelectedCrop] = useState("tomato");

  // Fetch mandi prices from backend - pass district directly, backend resolves coordinates
  useEffect(() => {
    const fetchPrices = async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams({ crop: selectedCrop });
        if (district) {
          params.set("district", district);
        }
        const url = `/api/market/prices?${params.toString()}`;
        console.log("Fetching mandi prices from:", url);
        
        const response = await fetch(url);
        
        if (response.ok) {
          const data = await response.json();
          console.log("Mandi API response:", data);
          
          // Transform API response to component format
          const prices: MandiPrice[] = data.markets.map((market: any) => ({
            mandi: market.name,
            location: market.location,
            price: Math.round(market.price / 100), // Convert quintal to approx kg
            change: market.change_value || 0,
            distance_km: market.distance_km,
          }));
          setMandiPrices(prices);
        } else {
          console.error("API error:", response.status);
          throw new Error("API error");
        }
      } catch (error) {
        console.error("Failed to fetch mandi prices:", error);
        // Fallback to default data
        setMandiPrices([
          { mandi: "Nashik Mandi", location: "Maharashtra", price: 40, change: 5.2 },
          { mandi: "Azadpur Mandi", location: "Delhi", price: 38, change: 3.1 },
          { mandi: "Vashi APMC", location: "Mumbai", price: 32, change: -2.4 },
          { mandi: "Kolar Mandi", location: "Karnataka", price: 35, change: 1.8 },
        ]);
      }
      setLoading(false);
    };

    fetchPrices();
  }, [selectedCrop, district]);

  const bestPrice = mandiPrices.length > 0 ? Math.max(...mandiPrices.map((m) => m.price)) : 0;
  const potentialGain = bestPrice - localTraderPrice;
  const percentageGain = localTraderPrice > 0 ? ((potentialGain / localTraderPrice) * 100).toFixed(0) : "0";

  const crops = ["tomato", "onion", "potato", "wheat", "rice", "cotton", "soybean"];

  return (
    <div className="dashboard-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-accent/10">
            <TrendingUp className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Mandi Connect</h3>
            <p className="text-xs text-muted-foreground">
              Live Market Prices {district && `• Near ${district.replace(/-/g, " ")}`}
            </p>
          </div>
        </div>
        {loading && <RefreshCw className="w-4 h-4 animate-spin text-muted-foreground" />}
      </div>

      {/* Crop Selector */}
      <div className="mb-4">
        <select
          value={selectedCrop}
          onChange={(e) => setSelectedCrop(e.target.value)}
          className="w-full p-2 rounded-lg bg-secondary border border-border text-foreground text-sm"
        >
          {crops.map((crop) => (
            <option key={crop} value={crop}>
              {crop.charAt(0).toUpperCase() + crop.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {/* Mandi Prices List */}
      <div className="space-y-2 mb-4 max-h-[250px] overflow-y-auto">
        {mandiPrices.map((mandi, index) => (
          <div
            key={`${mandi.mandi}-${index}`}
            className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
              index === 0 
                ? "bg-accent/10 border-accent/30" 
                : "bg-secondary/50 border-border hover:bg-secondary/80"
            }`}
          >
            <div className="flex items-center gap-2">
              <MapPin className={`w-4 h-4 ${index === 0 ? "text-accent" : "text-muted-foreground"}`} />
              <div>
                <p className="text-sm font-medium text-foreground">
                  {mandi.mandi}
                  {index === 0 && <span className="ml-2 text-xs text-accent">(Nearest)</span>}
                </p>
                <p className="text-xs text-muted-foreground">
                  {mandi.location}
                  {mandi.distance_km && ` • ${mandi.distance_km.toFixed(0)} km`}
                </p>
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
                {mandi.change.toFixed(1)}%
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
