 import {
   LineChart,
   Line,
   XAxis,
   YAxis,
   CartesianGrid,
   Tooltip,
   Legend,
   ResponsiveContainer,
 } from "recharts";
 
 // Generate mock sensor data for the last 24 hours
 const generateSensorData = () => {
   const data = [];
   const now = new Date();
   
   for (let i = 24; i >= 0; i--) {
     const time = new Date(now.getTime() - i * 60 * 60 * 1000);
     const hour = time.getHours().toString().padStart(2, "0") + ":00";
     
     // Simulate realistic sensor patterns
     const baseTemp = 26 + Math.sin(i / 4) * 4;
     const baseMoisture = 55 + Math.sin(i / 3) * 15 + Math.random() * 5;
     const baseHumidity = 65 + Math.cos(i / 5) * 10 + Math.random() * 5;
     
     data.push({
       time: hour,
       moisture: Math.round(baseMoisture),
       temperature: Math.round(baseTemp * 10) / 10,
       humidity: Math.round(baseHumidity),
     });
   }
   
   return data;
 };
 
 const sensorData = generateSensorData();
 
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="dashboard-card p-3 text-sm">
         <p className="text-muted-foreground mb-2">{label}</p>
         {payload.map((entry: any, index: number) => (
           <p key={index} style={{ color: entry.color }} className="flex justify-between gap-4">
             <span>{entry.name}:</span>
             <span className="font-medium">
               {entry.value}
               {entry.name === "Temperature" ? "°C" : "%"}
             </span>
           </p>
         ))}
       </div>
     );
   }
   return null;
 };
 
 export function SensorChart() {
   return (
    <div className="dashboard-card p-5">
       <h3 className="text-lg font-semibold text-foreground mb-4 font-display">
         24-Hour Sensor Readings
       </h3>
       
       <div className="chart-container h-72">
         <ResponsiveContainer width="100%" height="100%">
           <LineChart data={sensorData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
             <CartesianGrid 
               strokeDasharray="3 3" 
               stroke="hsl(160 25% 18%)" 
               vertical={false}
             />
             <XAxis 
               dataKey="time" 
               stroke="hsl(160 10% 45%)"
               tick={{ fill: "hsl(160 10% 55%)", fontSize: 11 }}
               tickLine={false}
               axisLine={{ stroke: "hsl(160 25% 18%)" }}
             />
             <YAxis 
               stroke="hsl(160 10% 45%)"
               tick={{ fill: "hsl(160 10% 55%)", fontSize: 11 }}
               tickLine={false}
               axisLine={{ stroke: "hsl(160 25% 18%)" }}
             />
             <Tooltip content={<CustomTooltip />} />
             <Legend 
               wrapperStyle={{ paddingTop: "20px" }}
               formatter={(value) => (
                 <span className="text-sm text-muted-foreground">{value}</span>
               )}
             />
             <Line
               type="monotone"
               dataKey="moisture"
               name="Soil Moisture (%)"
               stroke="hsl(199 89% 48%)"
               strokeWidth={2}
               dot={{ fill: "hsl(199 89% 48%)", strokeWidth: 0, r: 3 }}
               activeDot={{ r: 5, fill: "hsl(199 89% 48%)" }}
             />
             <Line
               type="monotone"
               dataKey="temperature"
               name="Temperature (°C)"
               stroke="hsl(25 95% 53%)"
               strokeWidth={2}
               dot={{ fill: "hsl(25 95% 53%)", strokeWidth: 0, r: 3 }}
               activeDot={{ r: 5, fill: "hsl(25 95% 53%)" }}
             />
             <Line
               type="monotone"
               dataKey="humidity"
               name="Humidity (%)"
               stroke="hsl(142 70% 45%)"
               strokeWidth={2}
               dot={{ fill: "hsl(142 70% 45%)", strokeWidth: 0, r: 3 }}
               activeDot={{ r: 5, fill: "hsl(142 70% 45%)" }}
             />
           </LineChart>
         </ResponsiveContainer>
       </div>
     </div>
   );
 }