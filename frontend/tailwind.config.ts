 import type { Config } from "tailwindcss";
 
 export default {
   darkMode: ["class"],
   content: ["./pages/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
   prefix: "",
   theme: {
     container: {
       center: true,
       padding: "2rem",
       screens: {
         "2xl": "1400px",
       },
     },
     extend: {
       fontFamily: {
         sans: ['Inter', 'system-ui', 'sans-serif'],
         display: ['Space Grotesk', 'Inter', 'sans-serif'],
       },
       colors: {
         border: "hsl(var(--border))",
         input: "hsl(var(--input))",
         ring: "hsl(var(--ring))",
         background: "hsl(var(--background))",
         foreground: "hsl(var(--foreground))",
         primary: {
           DEFAULT: "hsl(var(--primary))",
           foreground: "hsl(var(--primary-foreground))",
         },
         secondary: {
           DEFAULT: "hsl(var(--secondary))",
           foreground: "hsl(var(--secondary-foreground))",
         },
         destructive: {
           DEFAULT: "hsl(var(--destructive))",
           foreground: "hsl(var(--destructive-foreground))",
         },
         muted: {
           DEFAULT: "hsl(var(--muted))",
           foreground: "hsl(var(--muted-foreground))",
         },
         accent: {
           DEFAULT: "hsl(var(--accent))",
           foreground: "hsl(var(--accent-foreground))",
         },
         popover: {
           DEFAULT: "hsl(var(--popover))",
           foreground: "hsl(var(--popover-foreground))",
         },
         card: {
           DEFAULT: "hsl(var(--card))",
           foreground: "hsl(var(--card-foreground))",
         },
         sidebar: {
           DEFAULT: "hsl(var(--sidebar-background))",
           foreground: "hsl(var(--sidebar-foreground))",
           primary: "hsl(var(--sidebar-primary))",
           "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
           accent: "hsl(var(--sidebar-accent))",
           "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
           border: "hsl(var(--sidebar-border))",
           ring: "hsl(var(--sidebar-ring))",
         },
         // Extended Solarpunk Palette
         emerald: {
           deep: "hsl(var(--emerald-deep))",
           glow: "hsl(var(--emerald-glow))",
         },
         neon: {
           green: "hsl(var(--neon-green))",
           muted: "hsl(var(--neon-green-muted))",
         },
         slate: {
           dark: "hsl(var(--slate-dark))",
           medium: "hsl(var(--slate-medium))",
         },
         status: {
           good: "hsl(var(--status-good))",
           warning: "hsl(var(--status-warning))",
           danger: "hsl(var(--status-danger))",
           info: "hsl(var(--status-info))",
         },
         chart: {
           moisture: "hsl(var(--chart-moisture))",
           temperature: "hsl(var(--chart-temperature))",
           humidity: "hsl(var(--chart-humidity))",
         },
       },
       borderRadius: {
         lg: "var(--radius)",
         md: "calc(var(--radius) - 2px)",
         sm: "calc(var(--radius) - 4px)",
         xl: "calc(var(--radius) + 4px)",
         "2xl": "calc(var(--radius) + 8px)",
       },
       boxShadow: {
        'card': '0 2px 8px -2px hsl(160 30% 5% / 0.4)',
        'neon': '0 0 12px hsl(142 76% 55% / 0.2)',
       },
       keyframes: {
         "accordion-down": {
           from: { height: "0" },
           to: { height: "var(--radix-accordion-content-height)" },
         },
         "accordion-up": {
           from: { height: "var(--radix-accordion-content-height)" },
           to: { height: "0" },
         },
         "pulse-glow": {
           "0%, 100%": { boxShadow: "0 0 20px hsl(142 76% 55% / 0.4)" },
           "50%": { boxShadow: "0 0 30px hsl(142 76% 55% / 0.6)" },
         },
         "fade-in": {
           from: { opacity: "0", transform: "translateY(10px)" },
           to: { opacity: "1", transform: "translateY(0)" },
         },
         "slide-up": {
           from: { opacity: "0", transform: "translateY(20px)" },
           to: { opacity: "1", transform: "translateY(0)" },
         },
       },
       animation: {
         "accordion-down": "accordion-down 0.2s ease-out",
         "accordion-up": "accordion-up 0.2s ease-out",
         "pulse-glow": "pulse-glow 2s ease-in-out infinite",
         "fade-in": "fade-in 0.3s ease-out",
         "slide-up": "slide-up 0.4s ease-out",
       },
     },
   },
   plugins: [require("tailwindcss-animate")],
 } satisfies Config;
