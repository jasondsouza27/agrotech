import { useState, useEffect } from "react";
import { Sprout, Thermometer, Droplets, TestTube2, CloudRain, Leaf, RefreshCw, ChevronDown, ChevronUp, Sparkles } from "lucide-react";

interface CropRecommendation {
  crop: string;
  confidence: number;
  ideal_conditions: {
    N_range: string;
    P_range: string;
    K_range: string;
    temperature: string;
    humidity: string;
    pH: string;
    rainfall: string;
  };
}

interface SensorInput {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
  soil_moisture: number;
}

// Simulated sensor data (in production, this would come from actual IoT sensors)
const getSimulatedSensorData = (): SensorInput => ({
  nitrogen: Math.floor(Math.random() * 60) + 40,    // 40-100 kg/ha
  phosphorus: Math.floor(Math.random() * 50) + 30,  // 30-80 kg/ha
  potassium: Math.floor(Math.random() * 50) + 25,   // 25-75 kg/ha
  temperature: Math.floor(Math.random() * 15) + 20, // 20-35°C
  humidity: Math.floor(Math.random() * 40) + 50,    // 50-90%
  ph: parseFloat((Math.random() * 2 + 5.5).toFixed(1)), // 5.5-7.5
  rainfall: Math.floor(Math.random() * 150) + 50,   // 50-200 mm
  soil_moisture: Math.floor(Math.random() * 50) + 30, // 30-80%
});

export function CropRecommendation() {
  const [loading, setLoading] = useState(true);
  const [sensorData, setSensorData] = useState<SensorInput>(getSimulatedSensorData());
  const [recommendations, setRecommendations] = useState<CropRecommendation[]>([]);
  const [topCrop, setTopCrop] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [showAllRecs, setShowAllRecs] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRecommendation = async (data: SensorInput) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch("http://127.0.0.1:5000/api/crop/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        setTopCrop(result.top_recommendation);
        setConfidence(result.confidence);
        setRecommendations(result.all_recommendations || []);
      } else {
        throw new Error("Failed to get recommendation");
      }
    } catch (err) {
      console.error("Crop recommendation error:", err);
      setError("Unable to connect to prediction server");
      // Fallback recommendation
      setTopCrop("Maize");
      setConfidence(75);
      setRecommendations([
        { crop: "Maize", confidence: 75, ideal_conditions: { N_range: "60-100 kg/ha", P_range: "35-55 kg/ha", K_range: "30-50 kg/ha", temperature: "18-30°C", humidity: "50-75%", pH: "5.5-7.5", rainfall: "60-120 mm" } },
        { crop: "Rice", confidence: 65, ideal_conditions: { N_range: "80-120 kg/ha", P_range: "40-60 kg/ha", K_range: "40-60 kg/ha", temperature: "20-35°C", humidity: "80-95%", pH: "5.5-7.0", rainfall: "150-250 mm" } },
        { crop: "Wheat", confidence: 55, ideal_conditions: { N_range: "80-120 kg/ha", P_range: "30-50 kg/ha", K_range: "20-40 kg/ha", temperature: "10-25°C", humidity: "50-70%", pH: "6.0-7.5", rainfall: "50-100 mm" } },
      ]);
    }
    
    setLoading(false);
  };

  useEffect(() => {
    fetchRecommendation(sensorData);
  }, []);

  const handleRefresh = () => {
    const newData = getSimulatedSensorData();
    setSensorData(newData);
    fetchRecommendation(newData);
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 80) return "text-green-500";
    if (conf >= 60) return "text-yellow-500";
    return "text-orange-500";
  };

  const getCropEmoji = (crop: string): string => {
    const emojis: Record<string, string> = {
      "Rice": "🌾",
      "Wheat": "🌾",
      "Maize": "🌽",
      "Chickpea": "🫘",
      "Kidney Beans": "🫘",
      "Pigeon Peas": "🫘",
      "Moth Beans": "🫘",
      "Mung Bean": "🫛",
      "Black Gram": "🫘",
      "Lentil": "🫘",
      "Pomegranate": "🍎",
      "Banana": "🍌",
      "Mango": "🥭",
      "Grapes": "🍇",
      "Watermelon": "🍉",
      "Muskmelon": "🍈",
      "Apple": "🍎",
      "Orange": "🍊",
      "Papaya": "🍈",
      "Coconut": "🥥",
      "Cotton": "🧶",
      "Jute": "🌿",
      "Coffee": "☕",
    };
    return emojis[crop] || "🌱";
  };

  return (
    <div className="dashboard-card p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Sprout className="w-5 h-5 text-green-500" />
          <h3 className="text-sm font-medium text-muted-foreground font-display">
            AI Crop Recommendation
          </h3>
          <span className="text-xs bg-green-500/20 text-green-500 px-2 py-0.5 rounded-full">
            ML Powered
          </span>
        </div>
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="p-1.5 rounded-lg hover:bg-muted transition-colors"
          title="Refresh with new sensor data"
        >
          <RefreshCw className={`w-4 h-4 text-muted-foreground ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Current Sensor Readings */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <TestTube2 className="w-3 h-3" />
            <span>N</span>
          </div>
          <p className="text-sm font-medium">{sensorData.nitrogen}</p>
          <p className="text-xs text-muted-foreground">kg/ha</p>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <TestTube2 className="w-3 h-3" />
            <span>P</span>
          </div>
          <p className="text-sm font-medium">{sensorData.phosphorus}</p>
          <p className="text-xs text-muted-foreground">kg/ha</p>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <TestTube2 className="w-3 h-3" />
            <span>K</span>
          </div>
          <p className="text-sm font-medium">{sensorData.potassium}</p>
          <p className="text-xs text-muted-foreground">kg/ha</p>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <Leaf className="w-3 h-3" />
            <span>pH</span>
          </div>
          <p className="text-sm font-medium">{sensorData.ph}</p>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <Thermometer className="w-3 h-3" />
            <span>Temp</span>
          </div>
          <p className="text-sm font-medium">{sensorData.temperature}°C</p>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <Droplets className="w-3 h-3" />
            <span>Humidity</span>
          </div>
          <p className="text-sm font-medium">{sensorData.humidity}%</p>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <CloudRain className="w-3 h-3" />
            <span>Rain</span>
          </div>
          <p className="text-sm font-medium">{sensorData.rainfall}</p>
          <p className="text-xs text-muted-foreground">mm</p>
        </div>
        <div className="bg-muted/50 rounded-lg p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <Droplets className="w-3 h-3" />
            <span>Moisture</span>
          </div>
          <p className="text-sm font-medium">{sensorData.soil_moisture}%</p>
        </div>
      </div>

      {/* Top Recommendation */}
      {loading ? (
        <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-lg p-4 text-center">
          <div className="animate-pulse flex flex-col items-center">
            <div className="w-12 h-12 bg-green-500/20 rounded-full mb-2"></div>
            <div className="h-6 w-24 bg-green-500/20 rounded mb-1"></div>
            <div className="h-4 w-32 bg-green-500/20 rounded"></div>
          </div>
        </div>
      ) : (
        <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="text-4xl">{getCropEmoji(topCrop)}</div>
              <div>
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-yellow-500" />
                  <span className="text-xs text-muted-foreground">Recommended Crop</span>
                </div>
                <h4 className="text-xl font-bold text-foreground">{topCrop}</h4>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-muted-foreground">Confidence</p>
              <p className={`text-2xl font-bold ${getConfidenceColor(confidence)}`}>
                {confidence}%
              </p>
            </div>
          </div>
          
          {error && (
            <p className="text-xs text-orange-400 mt-2">
              ⚠️ {error} - showing cached recommendation
            </p>
          )}
        </div>
      )}

      {/* Show All Recommendations Toggle */}
      {recommendations.length > 1 && (
        <button
          onClick={() => setShowAllRecs(!showAllRecs)}
          className="w-full flex items-center justify-center gap-1 text-xs text-muted-foreground hover:text-foreground mt-3 py-2"
        >
          {showAllRecs ? (
            <>
              <ChevronUp className="w-3 h-3" />
              Hide alternatives
            </>
          ) : (
            <>
              <ChevronDown className="w-3 h-3" />
              Show {recommendations.length - 1} alternatives
            </>
          )}
        </button>
      )}

      {/* Alternative Recommendations */}
      {showAllRecs && (
        <div className="space-y-2 mt-2">
          {recommendations.slice(1).map((rec, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border border-border"
            >
              <div className="flex items-center gap-2">
                <span className="text-xl">{getCropEmoji(rec.crop)}</span>
                <div>
                  <p className="text-sm font-medium">{rec.crop}</p>
                  <p className="text-xs text-muted-foreground">
                    Temp: {rec.ideal_conditions.temperature}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className={`text-sm font-bold ${getConfidenceColor(rec.confidence)}`}>
                  {rec.confidence}%
                </p>
                <p className="text-xs text-muted-foreground">match</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-muted-foreground/60 mt-3 text-center">
        Prediction based on Random Forest ML model trained on 23 crops
      </p>
    </div>
  );
}
