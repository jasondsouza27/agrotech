 import { useState, useEffect } from "react";
 import { Sun, Cloud, Droplets, Wind, CloudRain, CloudLightning, CloudSnow, CloudFog, Loader2 } from "lucide-react";
 
 interface WeatherData {
   current: {
     temperature: number;
     feels_like: number;
     humidity: number;
     wind_speed: number;
     clouds: number;
     weather_main: string;
     weather_description: string;
     weather_icon: string;
     city: string;
   };
   forecast_24h: {
     text: string;
     temp_min: number;
     temp_max: number;
     rain_chance: number;
   };
 }
 
 const weatherIcons: Record<string, React.ReactNode> = {
   sun: <Sun className="w-10 h-10 text-status-warning" />,
   cloud: <Cloud className="w-10 h-10 text-muted-foreground" />,
   rain: <CloudRain className="w-10 h-10 text-status-info" />,
   storm: <CloudLightning className="w-10 h-10 text-status-warning" />,
   snow: <CloudSnow className="w-10 h-10 text-status-info" />,
   mist: <CloudFog className="w-10 h-10 text-muted-foreground" />,
 };
 
 const weatherIconsSmall: Record<string, React.ReactNode> = {
   sun: <Sun className="w-5 h-5 text-status-warning" />,
   cloud: <Cloud className="w-5 h-5 text-muted-foreground" />,
   rain: <CloudRain className="w-5 h-5 text-status-info" />,
   storm: <CloudLightning className="w-5 h-5 text-status-warning" />,
   snow: <CloudSnow className="w-5 h-5 text-status-info" />,
   mist: <CloudFog className="w-5 h-5 text-muted-foreground" />,
 };
 
 export function WeatherWidget({ district }: { district?: string }) {
   const [weather, setWeather] = useState<WeatherData | null>(null);
   const [loading, setLoading] = useState(true);
   const [error, setError] = useState<string | null>(null);
 
   useEffect(() => {
     const fetchWeather = async () => {
       try {
         setLoading(true);
         const params = district ? `?district=${encodeURIComponent(district)}` : "";
         const res = await fetch(`http://127.0.0.1:5000/api/weather${params}`);
         const data = await res.json();
         if (data.success) {
           setWeather(data);
           setError(null);
         } else {
           setError("Weather data unavailable");
         }
       } catch {
         setError("Cannot connect to weather service");
       } finally {
         setLoading(false);
       }
     };
     fetchWeather();
     // Refresh every 5 minutes
     const interval = setInterval(fetchWeather, 5 * 60 * 1000);
     return () => clearInterval(interval);
   }, [district]);
 
   if (loading) {
     return (
       <div className="dashboard-card p-5 flex items-center justify-center min-h-[200px]">
         <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
       </div>
     );
   }
 
   if (error || !weather) {
     return (
       <div className="dashboard-card p-5">
         <div className="flex items-center gap-2 mb-4">
           <Sun className="w-5 h-5 text-status-warning" />
           <h3 className="text-lg font-semibold text-foreground font-display">Current Weather</h3>
         </div>
         <p className="text-sm text-muted-foreground">{error || "No data"}</p>
       </div>
     );
   }
 
   const { current, forecast_24h } = weather;
   const icon = weatherIcons[current.weather_icon] || weatherIcons.sun;
   const headerIcon = weatherIconsSmall[current.weather_icon] || weatherIconsSmall.sun;
 
   return (
    <div className="dashboard-card p-5">
       <div className="flex items-center gap-2 mb-4">
         {headerIcon}
         <h3 className="text-lg font-semibold text-foreground font-display">
           Current Weather
         </h3>
         <span className="ml-auto text-xs text-muted-foreground">{current.city}</span>
       </div>
       
       <div className="flex items-center justify-between mb-4">
         <div className="p-3 rounded-xl bg-status-warning/10">
           {icon}
         </div>
         <div className="text-right">
           <span className="metric-value text-foreground">{Math.round(current.temperature)}°C</span>
           <p className="text-sm text-muted-foreground">{current.weather_description}</p>
           <p className="text-xs text-muted-foreground">Feels like {Math.round(current.feels_like)}°C</p>
         </div>
       </div>
       
       <div className="flex items-center gap-6 mb-4 text-sm">
         <div className="flex items-center gap-2">
           <Droplets className="w-4 h-4 text-status-info" />
           <span className="text-muted-foreground">Humidity:</span>
           <span className="text-foreground font-medium">{current.humidity}%</span>
         </div>
         <div className="flex items-center gap-2">
           <Wind className="w-4 h-4 text-muted-foreground" />
           <span className="text-muted-foreground">Wind:</span>
           <span className="text-foreground font-medium">{current.wind_speed} km/h</span>
         </div>
       </div>
       
       <div className="p-3 rounded-lg bg-secondary/50 border border-border">
         <div className="flex items-center justify-between mb-1">
           <p className="text-xs text-muted-foreground">24h Forecast</p>
           <p className="text-xs text-muted-foreground">
             {Math.round(forecast_24h.temp_min)}° / {Math.round(forecast_24h.temp_max)}°
           </p>
         </div>
         <p className="text-sm text-foreground">
           {forecast_24h.text}
         </p>
       </div>
     </div>
   );
 }