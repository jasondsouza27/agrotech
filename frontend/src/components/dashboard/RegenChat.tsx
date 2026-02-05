 import { useState } from "react";
 import { MessageCircle, Send, X, Leaf } from "lucide-react";
 import { Button } from "@/components/ui/button";
 import { Input } from "@/components/ui/input";
 
 interface Message {
   id: string;
   role: "user" | "assistant";
   content: string;
 }
 
 export function RegenChat() {
   const [isOpen, setIsOpen] = useState(false);
   const [input, setInput] = useState("");
   const [messages, setMessages] = useState<Message[]>([
     {
       id: "1",
       role: "assistant",
       content: "Hello! I'm Regen, your AI farming assistant. I have access to your live sensor data. How can I help you today?",
     },
   ]);
 
   const handleSend = () => {
     if (!input.trim()) return;
     
     const userMessage: Message = {
       id: Date.now().toString(),
       role: "user",
       content: input,
     };
     
     setMessages(prev => [...prev, userMessage]);
     setInput("");
     
     // Simulate AI response
     setTimeout(() => {
       const responses = [
         "Based on your current soil moisture at 52% and the warm temperature of 28°C, I'd recommend delaying irrigation until evening when evaporation rates are lower.",
         "Your sensors show optimal conditions. The humidity at 66% combined with good soil pH of 6.8 indicates healthy growing conditions for most crops.",
         "I notice your pump has been paused for efficiency. The AI auto-control is helping you save approximately 20 liters per hour during peak heat.",
       ];
       
       const assistantMessage: Message = {
         id: (Date.now() + 1).toString(),
         role: "assistant",
         content: responses[Math.floor(Math.random() * responses.length)],
       };
       
       setMessages(prev => [...prev, assistantMessage]);
     }, 1000);
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
           </div>
 
           {/* Input */}
           <div className="p-4 border-t border-border">
             <div className="flex gap-2">
               <Input
                 value={input}
                 onChange={(e) => setInput(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && handleSend()}
                 placeholder="Ask about your crops..."
                 className="flex-1 bg-secondary/50 border-border"
               />
               <Button
                 size="icon"
                 onClick={handleSend}
                className="bg-primary hover:bg-primary/90"
               >
                 <Send className="w-4 h-4" />
               </Button>
             </div>
           </div>
         </div>
       )}
     </>
   );
 }