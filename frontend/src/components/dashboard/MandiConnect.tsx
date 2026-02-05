import { useState, useEffect } from "react";
import { TrendingUp, MapPin, IndianRupee, ArrowUpRight, RefreshCw } from "lucide-react";
import { supabase } from "@/lib/supabase";

interface MandiPrice {
  mandi: string;
  location: string;
  price: number;
  change: number;
  distance_km?: number;
}

// District coordinates for proximity sorting - matches Signup district labels
const DISTRICT_COORDINATES: Record<string, { lat: number; lon: number }> = {
  // Maharashtra
  "nashik": { lat: 19.9975, lon: 73.7898 },
  "pune": { lat: 18.5204, lon: 73.8567 },
  "nagpur": { lat: 21.1458, lon: 79.0882 },
  "ahmednagar": { lat: 19.0948, lon: 74.7480 },
  "solapur": { lat: 17.6599, lon: 75.9064 },
  "kolhapur": { lat: 16.7050, lon: 74.2433 },
  "sangli": { lat: 16.8524, lon: 74.5815 },
  "satara": { lat: 17.6805, lon: 74.0183 },
  "aurangabad": { lat: 19.8762, lon: 75.3433 },
  "jalgaon": { lat: 21.0077, lon: 75.5626 },
  // Karnataka
  "bangalore rural": { lat: 13.2257, lon: 77.5750 },
  "bangalore-rural": { lat: 13.2257, lon: 77.5750 },
  "belgaum": { lat: 15.8497, lon: 74.4977 },
  "bellary": { lat: 15.1394, lon: 76.9214 },
  "bidar": { lat: 17.9104, lon: 77.5199 },
  "dharwad": { lat: 15.4589, lon: 75.0078 },
  "gulbarga": { lat: 17.3297, lon: 76.8343 },
  "hassan": { lat: 13.0068, lon: 76.1004 },
  "kolar": { lat: 13.1360, lon: 78.1292 },
  "mandya": { lat: 12.5218, lon: 76.8951 },
  "mysore": { lat: 12.2958, lon: 76.6394 },
  "shimoga": { lat: 13.9299, lon: 75.5681 },
  "tumkur": { lat: 13.3379, lon: 77.1173 },
  // Gujarat
  "ahmedabad": { lat: 23.0225, lon: 72.5714 },
  "amreli": { lat: 21.6032, lon: 71.2216 },
  "anand": { lat: 22.5645, lon: 72.9289 },
  "banaskantha": { lat: 24.1722, lon: 72.4350 },
  "bharuch": { lat: 21.7051, lon: 72.9959 },
  "bhavnagar": { lat: 21.7645, lon: 72.1519 },
  "junagadh": { lat: 21.5222, lon: 70.4579 },
  "kheda": { lat: 22.7507, lon: 72.6847 },
  "mehsana": { lat: 23.5880, lon: 72.3693 },
  "rajkot": { lat: 22.3039, lon: 70.8022 },
  "surat": { lat: 21.1702, lon: 72.8311 },
  "vadodara": { lat: 22.3072, lon: 73.1812 },
  // Punjab
  "amritsar": { lat: 31.6340, lon: 74.8723 },
  "bathinda": { lat: 30.2110, lon: 74.9455 },
  "ferozepur": { lat: 30.9331, lon: 74.6225 },
  "gurdaspur": { lat: 32.0414, lon: 75.4031 },
  "jalandhar": { lat: 31.3260, lon: 75.5762 },
  "ludhiana": { lat: 30.9010, lon: 75.8573 },
  "moga": { lat: 30.8162, lon: 75.1741 },
  "patiala": { lat: 30.3398, lon: 76.3869 },
  "sangrur": { lat: 30.2506, lon: 75.8442 },
  // Haryana
  "ambala": { lat: 30.3782, lon: 76.7767 },
  "hisar": { lat: 29.1492, lon: 75.7217 },
  "karnal": { lat: 29.6857, lon: 76.9905 },
  "kurukshetra": { lat: 29.9695, lon: 76.8783 },
  "rohtak": { lat: 28.8955, lon: 76.6066 },
  "sirsa": { lat: 29.5349, lon: 75.0288 },
  "sonipat": { lat: 28.9288, lon: 77.0913 },
  // Uttar Pradesh
  "agra": { lat: 27.1767, lon: 78.0081 },
  "aligarh": { lat: 27.8974, lon: 78.0880 },
  "allahabad": { lat: 25.4358, lon: 81.8463 },
  "bareilly": { lat: 28.3670, lon: 79.4304 },
  "gorakhpur": { lat: 26.7606, lon: 83.3732 },
  "jhansi": { lat: 25.4484, lon: 78.5685 },
  "kanpur": { lat: 26.4499, lon: 80.3319 },
  "lucknow": { lat: 26.8467, lon: 80.9462 },
  "mathura": { lat: 27.4924, lon: 77.6737 },
  "meerut": { lat: 28.9845, lon: 77.7064 },
  "moradabad": { lat: 28.8386, lon: 78.7733 },
  "muzaffarnagar": { lat: 29.4727, lon: 77.7085 },
  "varanasi": { lat: 25.3176, lon: 82.9739 },
  // Madhya Pradesh
  "bhopal": { lat: 23.2599, lon: 77.4126 },
  "gwalior": { lat: 26.2183, lon: 78.1828 },
  "indore": { lat: 22.7196, lon: 75.8577 },
  "jabalpur": { lat: 23.1815, lon: 79.9864 },
  "rewa": { lat: 24.5373, lon: 81.3042 },
  "sagar": { lat: 23.8388, lon: 78.7378 },
  "ujjain": { lat: 23.1765, lon: 75.7885 },
  // Rajasthan
  "ajmer": { lat: 26.4499, lon: 74.6399 },
  "alwar": { lat: 27.5530, lon: 76.6346 },
  "bikaner": { lat: 28.0229, lon: 73.3119 },
  "jaipur": { lat: 26.9124, lon: 75.7873 },
  "jodhpur": { lat: 26.2389, lon: 73.0243 },
  "kota": { lat: 25.2138, lon: 75.8648 },
  "sikar": { lat: 27.6094, lon: 75.1398 },
  "udaipur": { lat: 24.5854, lon: 73.7125 },
  // Tamil Nadu
  "chennai": { lat: 13.0827, lon: 80.2707 },
  "coimbatore": { lat: 11.0168, lon: 76.9558 },
  "erode": { lat: 11.3410, lon: 77.7172 },
  "madurai": { lat: 9.9252, lon: 78.1198 },
  "salem": { lat: 11.6643, lon: 78.1460 },
  "thanjavur": { lat: 10.7870, lon: 79.1378 },
  "tiruchirappalli": { lat: 10.7905, lon: 78.7047 },
  "tirunelveli": { lat: 8.7139, lon: 77.7567 },
  // Andhra Pradesh / Telangana
  "anantapur": { lat: 14.6819, lon: 77.6006 },
  "guntur": { lat: 16.3067, lon: 80.4365 },
  "hyderabad": { lat: 17.3850, lon: 78.4867 },
  "karimnagar": { lat: 18.4386, lon: 79.1288 },
  "krishna": { lat: 16.6100, lon: 80.7214 },
  "kurnool": { lat: 15.8281, lon: 78.0373 },
  "nalgonda": { lat: 17.0575, lon: 79.2690 },
  "warangal": { lat: 17.9784, lon: 79.5941 },
  "visakhapatnam": { lat: 17.6868, lon: 83.2185 },
  // West Bengal
  "barddhaman": { lat: 23.2324, lon: 87.8615 },
  "hooghly": { lat: 22.9086, lon: 88.3966 },
  "kolkata": { lat: 22.5726, lon: 88.3639 },
  "murshidabad": { lat: 24.1750, lon: 88.2750 },
  "nadia": { lat: 23.4710, lon: 88.5565 },
  // Bihar
  "bhagalpur": { lat: 25.2425, lon: 86.9842 },
  "darbhanga": { lat: 26.1542, lon: 85.8918 },
  "gaya": { lat: 24.7955, lon: 85.0002 },
  "muzaffarpur": { lat: 26.1225, lon: 85.3906 },
  "patna": { lat: 25.5941, lon: 85.1376 },
  // Odisha
  "balasore": { lat: 21.4934, lon: 86.9135 },
  "cuttack": { lat: 20.4625, lon: 85.8830 },
  "ganjam": { lat: 19.5860, lon: 84.6897 },
  "puri": { lat: 19.8135, lon: 85.8312 },
  // Kerala
  "alappuzha": { lat: 9.4981, lon: 76.3388 },
  "ernakulam": { lat: 9.9816, lon: 76.2999 },
  "kozhikode": { lat: 11.2588, lon: 75.7804 },
  "palakkad": { lat: 10.7867, lon: 76.6548 },
  "thrissur": { lat: 10.5276, lon: 76.2144 },
  // Default (Mumbai)
  "default": { lat: 19.0760, lon: 72.8777 },
};

const localTraderPrice = 15;

export function MandiConnect() {
  const [mandiPrices, setMandiPrices] = useState<MandiPrice[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCrop, setSelectedCrop] = useState("tomato");
  const [userDistrict, setUserDistrict] = useState<string | null>(null);
  const [userCoords, setUserCoords] = useState<{ lat: number; lon: number } | null>(null);

  // Get user's district from Supabase
  useEffect(() => {
    const getUserLocation = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        console.log("Supabase user:", user);
        console.log("User metadata:", user?.user_metadata);
        
        if (user?.user_metadata?.district) {
          const district = user.user_metadata.district;
          // Convert district name to key format (e.g., "Nashik" -> "nashik", "Bangalore Rural" -> "bangalore rural")
          const districtKey = district.toLowerCase().trim();
          
          console.log("District from profile:", district, "-> key:", districtKey);
          setUserDistrict(district); // Store original name for display
          
          // Look up coordinates - try exact match first, then with hyphen
          let coords = DISTRICT_COORDINATES[districtKey];
          if (!coords) {
            // Try with hyphen instead of space
            coords = DISTRICT_COORDINATES[districtKey.replace(/\s+/g, "-")];
          }
          if (!coords) {
            console.log("No coordinates found for district, using default");
            coords = DISTRICT_COORDINATES["default"];
          }
          
          console.log("Coordinates for", districtKey, ":", coords);
          setUserCoords(coords);
        } else {
          console.log("No district in user metadata, using default");
          setUserCoords(DISTRICT_COORDINATES["default"]);
        }
      } catch (error) {
        console.error("Error getting user location:", error);
        setUserCoords(DISTRICT_COORDINATES["default"]);
      }
    };
    getUserLocation();
  }, []);

  // Fetch mandi prices from backend
  useEffect(() => {
    const fetchPrices = async () => {
      // Wait for coordinates to be set
      if (!userCoords) return;
      
      setLoading(true);
      try {
        const url = `http://127.0.0.1:5000/api/market/prices?crop=${selectedCrop}&lat=${userCoords.lat}&lon=${userCoords.lon}`;
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
  }, [selectedCrop, userCoords]);

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
              Live Market Prices {userDistrict && `• Near ${userDistrict.replace(/-/g, " ")}`}
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
