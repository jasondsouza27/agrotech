 import { useState, useRef } from "react";
 import { Leaf, Upload, X, CheckCircle, AlertTriangle, Info } from "lucide-react";
 import { Button } from "@/components/ui/button";
 import {
   Dialog,
   DialogContent,
   DialogHeader,
   DialogTitle,
   DialogTrigger,
 } from "@/components/ui/dialog";

// Backend API URL - change this for production
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

interface ScanResult {
  status: "healthy" | "warning" | "info";
  diagnosis: string;
  recommendation: string;
  confidence?: number;
}

 export function LeafDoctor() {
   const [isOpen, setIsOpen] = useState(false);
   const [scanning, setScanning] = useState(false);
   const [result, setResult] = useState<null | ScanResult>(null);
   const [error, setError] = useState<string | null>(null);
   const fileInputRef = useRef<HTMLInputElement>(null);

   const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
     const file = event.target.files?.[0];
     if (!file) return;

     setScanning(true);
     setError(null);
     setResult(null);

     try {
       const formData = new FormData();
       formData.append("image", file);

       const response = await fetch(`${API_BASE_URL}/api/scan/image`, {
         method: "POST",
         body: formData,
       });

       const data = await response.json();

       if (data.success) {
         // Map backend response to frontend format
         const resultData = data.result;
         setResult({
           status: resultData.status === "healthy" ? "healthy" : 
                   resultData.status === "info" ? "info" : "warning",
           diagnosis: `${resultData.diagnosis}: ${resultData.description}`,
           recommendation: resultData.remedy,
           confidence: resultData.confidence,
         });
       } else {
         setError(data.message || "Failed to analyze image");
       }
     } catch (err) {
       console.error("Scan error:", err);
       setError("Could not connect to the server. Make sure the backend is running on port 5000.");
     } finally {
       setScanning(false);
     }
   };

   const handleScanClick = () => {
     fileInputRef.current?.click();
   };

   const resetScan = () => {
     setResult(null);
     setScanning(false);
     setError(null);
     if (fileInputRef.current) {
       fileInputRef.current.value = "";
     }
   };

   const getStatusIcon = () => {
     if (!result) return null;
     if (result.status === "healthy") return <CheckCircle className="w-5 h-5 text-status-good" />;
     if (result.status === "info") return <Info className="w-5 h-5 text-blue-500" />;
     return <AlertTriangle className="w-5 h-5 text-status-warning" />;
   };

   const getStatusColor = () => {
     if (!result) return "";
     if (result.status === "healthy") return "text-status-good";
     if (result.status === "info") return "text-blue-500";
     return "text-status-warning";
   };

   const getStatusBg = () => {
     if (!result) return "";
     if (result.status === "healthy") return "bg-status-good/10 border-status-good/30";
     if (result.status === "info") return "bg-blue-500/10 border-blue-500/30";
     return "bg-status-warning/10 border-status-warning/30";
   };

   return (
    <div className="dashboard-card p-5">
       <div className="flex items-center gap-2 mb-3">
        <Leaf className="w-5 h-5 text-accent" />
         <h3 className="text-lg font-semibold text-foreground font-display">
           Leaf Doctor
         </h3>
       </div>
       
       <p className="text-sm text-muted-foreground mb-4">
         AI-powered crop health analysis. Upload a leaf image for instant diagnosis.
       </p>
       
       <Dialog open={isOpen} onOpenChange={setIsOpen}>
         <DialogTrigger asChild>
          <Button className="w-full bg-primary hover:bg-primary/90 transition-all">
             <Upload className="w-4 h-4 mr-2" />
             Scan Leaf
           </Button>
         </DialogTrigger>
        <DialogContent className="border-border bg-card max-w-md">
           <DialogHeader>
             <DialogTitle className="flex items-center gap-2 font-display">
              <Leaf className="w-5 h-5 text-accent" />
               Leaf Health Scanner
             </DialogTitle>
           </DialogHeader>
           
           {/* Hidden file input */}
           <input
             type="file"
             ref={fileInputRef}
             onChange={handleFileSelect}
             accept="image/*"
             className="hidden"
           />

           {error && (
             <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
               {error}
             </div>
           )}

           {!result ? (
             <div className="space-y-4">
               <div 
                 className="border-2 border-dashed border-border rounded-xl p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                 onClick={handleScanClick}
               >
                 {scanning ? (
                   <div className="space-y-3">
                    <div className="w-12 h-12 mx-auto rounded-full border-2 border-accent border-t-transparent animate-spin" />
                     <p className="text-sm text-muted-foreground">Analyzing leaf pattern...</p>
                   </div>
                 ) : (
                   <>
                     <Upload className="w-10 h-10 mx-auto text-muted-foreground mb-3" />
                     <p className="text-sm text-foreground font-medium mb-1">
                       Click to upload leaf image
                     </p>
                     <p className="text-xs text-muted-foreground">
                       JPG, PNG up to 10MB
                     </p>
                   </>
                 )}
               </div>
               
               <p className="text-xs text-muted-foreground text-center">
                 Our AI can detect 50+ crop diseases with 95% accuracy
               </p>
             </div>
           ) : (
             <div className="space-y-4">
               <div className={`p-4 rounded-xl border ${getStatusBg()}`}>
                 <div className="flex items-center gap-2 mb-2">
                   {getStatusIcon()}
                   <span className={`font-medium ${getStatusColor()}`}>
                     {result.status === "healthy" ? "Healthy" : 
                      result.status === "info" ? "Detection Info" : "Issue Detected"}
                   </span>
                   {result.confidence && (
                     <span className="ml-auto text-xs text-muted-foreground">
                       {Math.round(result.confidence * 100)}% confidence
                     </span>
                   )}
                 </div>
                 <p className="text-sm text-foreground">{result.diagnosis}</p>
               </div>
               
               <div className="p-4 rounded-xl bg-secondary/50 border border-border">
                 <p className="text-xs text-muted-foreground mb-1">Recommendation</p>
                 <p className="text-sm text-foreground">{result.recommendation}</p>
               </div>
               
               <Button 
                 variant="outline" 
                 className="w-full"
                 onClick={resetScan}
               >
                 Scan Another Leaf
               </Button>
             </div>
           )}
         </DialogContent>
       </Dialog>
     </div>
   );
 }