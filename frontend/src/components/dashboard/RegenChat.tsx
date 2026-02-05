 import { useState } from "react";
import { MessageCircle, Send, X, Leaf, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const API_URL = "http://127.0.0.1:5000";

export function RegenChat() {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hello! I'm Regen, your AI farming assistant powered by Llama 3.1. I have access to your live sensor data. How can I help you today?",
    },
  ]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    
    try {
      // Get conversation history (exclude initial greeting)
      const history = messages.slice(1).map(m => ({
        role: m.role,
        content: m.content
      }));
      
      // Mock sensor data - in production, get from actual sensors
      const sensorData = {
        soil_moisture: 52,
        temperature: 28,
        humidity: 66,
        soil_ph: 6.8,
        nitrogen: 45,
        phosphorus: 38,
        potassium: 42
      };
      
      const response = await fetch(`${API_URL}/api/llm/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input,
          history: history,
          sensor_data: sensorData
        }),
      });
      
      const data = await response.json();
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || "I couldn't process that. Please try again.",
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I'm having trouble connecting. Please make sure LM Studio is running with the Llama 3.1 model loaded.",
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
     <>
       {/* Floating Button */}
       <button
         onClick={() => setIsOpen(true)}
          className={`chat-float w-14 h-14 rounded-full bg-primary flex items-center justify-center transition-all hover:scale-105 hover:bg-primary/90 ${
           isOpen ? "opacity-0 pointer-events-none" : "opacity-100"
         }`}
       >
         <MessageCircle className="w-6 h-6 text-primary-foreground" />
       </button>
 
       {/* Chat Window */}
       {isOpen && (
          <div className="fixed bottom-6 right-6 z-50 w-80 sm:w-96 dashboard-card animate-slide-up">
           {/* Header */}
           <div className="flex items-center justify-between p-4 border-b border-border">
             <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                 <Leaf className="w-4 h-4 text-primary-foreground" />
               </div>
               <div>
                 <h4 className="text-sm font-semibold text-foreground font-display">
                   Regen AI
                 </h4>
                  <p className="text-xs text-accent flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                   Online • Live Data Access
                 </p>
               </div>
             </div>
             <button
               onClick={() => setIsOpen(false)}
               className="p-1.5 rounded-lg hover:bg-secondary/50 transition-colors"
             >
               <X className="w-4 h-4 text-muted-foreground" />
             </button>
           </div>
 
           {/* Messages */}
           <div className="h-72 overflow-y-auto p-4 space-y-3 scrollbar-thin">
             {messages.map((message) => (
               <div
                 key={message.id}
                 className={`flex ${
                   message.role === "user" ? "justify-end" : "justify-start"
                 }`}
               >
                 <div
                   className={`max-w-[85%] p-3 rounded-xl text-sm ${
                     message.role === "user"
                       ? "bg-primary text-primary-foreground"
                       : "bg-secondary/70 text-foreground border border-border"
                   }`}
                 >
                   {message.content}
                 </div>
               </div>
             ))}
             {isLoading && (
               <div className="flex justify-start">
                 <div className="bg-secondary/70 text-foreground border border-border p-3 rounded-xl text-sm flex items-center gap-2">
                   <Loader2 className="w-4 h-4 animate-spin" />
                   Thinking...
                 </div>
               </div>
             )}
           </div>

           {/* Input */}
           <div className="p-4 border-t border-border">
             <div className="flex gap-2">
               <Input
                 value={input}
                 onChange={(e) => setInput(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && !isLoading && handleSend()}
                 placeholder="Ask about your crops..."
                 className="flex-1 bg-secondary/50 border-border"
                 disabled={isLoading}
               />
               <Button
                 size="icon"
                 onClick={handleSend}
                 disabled={isLoading}
                 className="bg-primary hover:bg-primary/90"
               >
                 {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
               </Button>
             </div>
           </div>
         </div>
       )}
     </>
   );
 }