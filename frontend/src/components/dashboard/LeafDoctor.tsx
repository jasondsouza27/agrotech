 import { useState, useRef, useCallback, useEffect } from "react";
 import { Leaf, Upload, X, CheckCircle, AlertTriangle, Info, Camera, ImageIcon, Video } from "lucide-react";
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

type ScanMode = "choose" | "upload" | "camera";

interface ScanResult {
  status: "healthy" | "warning" | "info";
  diagnosis: string;
  recommendation: string;
  confidence?: number;
}

 export function LeafDoctor() {
   const [isOpen, setIsOpen] = useState(false);
   const [mode, setMode] = useState<ScanMode>("choose");
   const [scanning, setScanning] = useState(false);
   const [result, setResult] = useState<null | ScanResult>(null);
   const [error, setError] = useState<string | null>(null);
   const [cameraActive, setCameraActive] = useState(false);
   const fileInputRef = useRef<HTMLInputElement>(null);
   const videoRef = useRef<HTMLVideoElement>(null);
   const canvasRef = useRef<HTMLCanvasElement>(null);
   const streamRef = useRef<MediaStream | null>(null);

   // Cleanup camera on unmount or dialog close
   const stopCamera = useCallback(() => {
     if (streamRef.current) {
       streamRef.current.getTracks().forEach(track => track.stop());
       streamRef.current = null;
     }
     setCameraActive(false);
   }, []);

   useEffect(() => {
     if (!isOpen) {
       stopCamera();
       setMode("choose");
     }
   }, [isOpen, stopCamera]);

   // Start webcam
   const startCamera = async () => {
     setError(null);
     try {
       const stream = await navigator.mediaDevices.getUserMedia({
         video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } }
       });
       streamRef.current = stream;
       if (videoRef.current) {
         videoRef.current.srcObject = stream;
         await videoRef.current.play();
       }
       setCameraActive(true);
     } catch (err) {
       console.error("Camera error:", err);
       setError("Could not access camera. Please allow camera permission or try the upload option.");
     }
   };

   // Capture frame from webcam and send to backend
   const captureFrame = async () => {
     if (!videoRef.current || !canvasRef.current) return;

     setScanning(true);
     setError(null);

     try {
       const video = videoRef.current;
       const canvas = canvasRef.current;
       canvas.width = video.videoWidth;
       canvas.height = video.videoHeight;
       const ctx = canvas.getContext("2d");
       if (!ctx) return;

       ctx.drawImage(video, 0, 0);
       const frameDataUrl = canvas.toDataURL("image/jpeg", 0.85);

       const response = await fetch(`${API_BASE_URL}/api/scan/frame`, {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ frame: frameDataUrl }),
       });

       const data = await response.json();

       if (data.success) {
         const resultData = data.result;
         stopCamera();
         setResult({
           status: resultData.status === "healthy" ? "healthy" :
                   resultData.status === "info" ? "info" : "warning",
           diagnosis: `${resultData.diagnosis}: ${resultData.description}`,
           recommendation: resultData.remedy,
           confidence: resultData.confidence,
         });
       } else {
         setError(data.message || "Failed to analyze frame");
       }
     } catch (err) {
       console.error("Capture error:", err);
       setError("Could not connect to the server. Make sure the backend is running on port 5000.");
     } finally {
       setScanning(false);
     }
   };

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
     setMode("choose");
     stopCamera();
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
         AI-powered crop health analysis. Upload a leaf image or use your camera for instant diagnosis.
       </p>
       
       <Dialog open={isOpen} onOpenChange={setIsOpen}>
         <DialogTrigger asChild>
          <Button className="w-full bg-primary hover:bg-primary/90 transition-all">
             <Leaf className="w-4 h-4 mr-2" />
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
           {/* Hidden canvas for frame capture */}
           <canvas ref={canvasRef} className="hidden" />

           {error && (
             <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
               {error}
             </div>
           )}

           {/* Mode Chooser */}
           {mode === "choose" && !result && (
             <div className="space-y-4">
               <p className="text-sm text-muted-foreground text-center">Choose a scan method</p>
               <div className="grid grid-cols-2 gap-3">
                 <button
                   className="flex flex-col items-center gap-3 p-6 rounded-xl border-2 border-dashed border-border hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer"
                   onClick={() => { setMode("upload"); setTimeout(() => handleScanClick(), 100); }}
                 >
                   <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                     <ImageIcon className="w-6 h-6 text-primary" />
                   </div>
                   <div className="text-center">
                     <p className="text-sm font-medium text-foreground">Upload Image</p>
                     <p className="text-xs text-muted-foreground mt-1">JPG, PNG up to 10MB</p>
                   </div>
                 </button>
                 <button
                   className="flex flex-col items-center gap-3 p-6 rounded-xl border-2 border-dashed border-border hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer"
                   onClick={() => { setMode("camera"); startCamera(); }}
                 >
                   <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                     <Video className="w-6 h-6 text-primary" />
                   </div>
                   <div className="text-center">
                     <p className="text-sm font-medium text-foreground">Live Camera</p>
                     <p className="text-xs text-muted-foreground mt-1">Real-time webcam scan</p>
                   </div>
                 </button>
               </div>
               <p className="text-xs text-muted-foreground text-center">
                 Our AI detects 50+ crop diseases with 95% accuracy
               </p>
             </div>
           )}

           {/* Upload Mode */}
           {mode === "upload" && !result && (
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
               <Button variant="ghost" className="w-full text-muted-foreground" onClick={resetScan}>
                 ← Back to scan options
               </Button>
             </div>
           )}

           {/* Camera Mode */}
           {mode === "camera" && !result && (
             <div className="space-y-4">
               <div className="relative rounded-xl overflow-hidden bg-black aspect-[4/3]">
                 <video
                   ref={videoRef}
                   autoPlay
                   playsInline
                   muted
                   className="w-full h-full object-cover"
                 />
                 {/* Center guide box - shows where to hold the leaf */}
                 {cameraActive && !scanning && (
                   <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                     <div className="w-[50%] h-[60%] border-2 border-dashed border-green-400/70 rounded-xl flex items-center justify-center">
                       <span className="text-green-400/80 text-xs bg-black/40 px-2 py-1 rounded">Place leaf here</span>
                     </div>
                   </div>
                 )}
                 {!cameraActive && !error && (
                   <div className="absolute inset-0 flex items-center justify-center">
                     <div className="w-10 h-10 rounded-full border-2 border-white border-t-transparent animate-spin" />
                   </div>
                 )}
                 {/* Scanning overlay */}
                 {scanning && (
                   <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                     <div className="text-center text-white">
                       <div className="w-10 h-10 mx-auto rounded-full border-2 border-white border-t-transparent animate-spin mb-2" />
                       <p className="text-sm">Analyzing...</p>
                     </div>
                   </div>
                 )}
               </div>
               <p className="text-xs text-muted-foreground text-center">
                 Hold the leaf in front of the camera, then capture
               </p>
               <div className="flex gap-2">
                 <Button variant="ghost" className="flex-1 text-muted-foreground" onClick={resetScan}>
                   ← Back
                 </Button>
                 <Button
                   className="flex-1 bg-primary hover:bg-primary/90"
                   onClick={captureFrame}
                   disabled={!cameraActive || scanning}
                 >
                   <Camera className="w-4 h-4 mr-2" />
                   {scanning ? "Analyzing..." : "Capture & Analyze"}
                 </Button>
               </div>
             </div>
           )}

           {/* Results (same for both modes) */}
           {result && (
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