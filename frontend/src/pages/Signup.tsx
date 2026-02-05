import { useState, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Leaf, User, Mail, Lock, MapPin, UserPlus, Check, ChevronsUpDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabase";
import { useToast } from "@/hooks/use-toast";

// Indian Districts List (major agricultural districts)
const DISTRICTS = [
  // Maharashtra
  { value: "nashik", label: "Nashik", state: "Maharashtra" },
  { value: "pune", label: "Pune", state: "Maharashtra" },
  { value: "nagpur", label: "Nagpur", state: "Maharashtra" },
  { value: "ahmednagar", label: "Ahmednagar", state: "Maharashtra" },
  { value: "solapur", label: "Solapur", state: "Maharashtra" },
  { value: "kolhapur", label: "Kolhapur", state: "Maharashtra" },
  { value: "sangli", label: "Sangli", state: "Maharashtra" },
  { value: "satara", label: "Satara", state: "Maharashtra" },
  { value: "aurangabad", label: "Aurangabad", state: "Maharashtra" },
  { value: "jalgaon", label: "Jalgaon", state: "Maharashtra" },
  // Karnataka
  { value: "bangalore-rural", label: "Bangalore Rural", state: "Karnataka" },
  { value: "belgaum", label: "Belgaum", state: "Karnataka" },
  { value: "bellary", label: "Bellary", state: "Karnataka" },
  { value: "bidar", label: "Bidar", state: "Karnataka" },
  { value: "dharwad", label: "Dharwad", state: "Karnataka" },
  { value: "gulbarga", label: "Gulbarga", state: "Karnataka" },
  { value: "hassan", label: "Hassan", state: "Karnataka" },
  { value: "kolar", label: "Kolar", state: "Karnataka" },
  { value: "mandya", label: "Mandya", state: "Karnataka" },
  { value: "mysore", label: "Mysore", state: "Karnataka" },
  { value: "shimoga", label: "Shimoga", state: "Karnataka" },
  { value: "tumkur", label: "Tumkur", state: "Karnataka" },
  // Gujarat
  { value: "ahmedabad", label: "Ahmedabad", state: "Gujarat" },
  { value: "amreli", label: "Amreli", state: "Gujarat" },
  { value: "anand", label: "Anand", state: "Gujarat" },
  { value: "banaskantha", label: "Banaskantha", state: "Gujarat" },
  { value: "bharuch", label: "Bharuch", state: "Gujarat" },
  { value: "bhavnagar", label: "Bhavnagar", state: "Gujarat" },
  { value: "junagadh", label: "Junagadh", state: "Gujarat" },
  { value: "kheda", label: "Kheda", state: "Gujarat" },
  { value: "mehsana", label: "Mehsana", state: "Gujarat" },
  { value: "rajkot", label: "Rajkot", state: "Gujarat" },
  { value: "surat", label: "Surat", state: "Gujarat" },
  { value: "vadodara", label: "Vadodara", state: "Gujarat" },
  // Punjab
  { value: "amritsar", label: "Amritsar", state: "Punjab" },
  { value: "bathinda", label: "Bathinda", state: "Punjab" },
  { value: "ferozepur", label: "Ferozepur", state: "Punjab" },
  { value: "gurdaspur", label: "Gurdaspur", state: "Punjab" },
  { value: "jalandhar", label: "Jalandhar", state: "Punjab" },
  { value: "ludhiana", label: "Ludhiana", state: "Punjab" },
  { value: "moga", label: "Moga", state: "Punjab" },
  { value: "patiala", label: "Patiala", state: "Punjab" },
  { value: "sangrur", label: "Sangrur", state: "Punjab" },
  // Haryana
  { value: "ambala", label: "Ambala", state: "Haryana" },
  { value: "hisar", label: "Hisar", state: "Haryana" },
  { value: "karnal", label: "Karnal", state: "Haryana" },
  { value: "kurukshetra", label: "Kurukshetra", state: "Haryana" },
  { value: "rohtak", label: "Rohtak", state: "Haryana" },
  { value: "sirsa", label: "Sirsa", state: "Haryana" },
  { value: "sonipat", label: "Sonipat", state: "Haryana" },
  // Uttar Pradesh
  { value: "agra", label: "Agra", state: "Uttar Pradesh" },
  { value: "aligarh", label: "Aligarh", state: "Uttar Pradesh" },
  { value: "allahabad", label: "Allahabad", state: "Uttar Pradesh" },
  { value: "bareilly", label: "Bareilly", state: "Uttar Pradesh" },
  { value: "gorakhpur", label: "Gorakhpur", state: "Uttar Pradesh" },
  { value: "jhansi", label: "Jhansi", state: "Uttar Pradesh" },
  { value: "kanpur", label: "Kanpur", state: "Uttar Pradesh" },
  { value: "lucknow", label: "Lucknow", state: "Uttar Pradesh" },
  { value: "mathura", label: "Mathura", state: "Uttar Pradesh" },
  { value: "meerut", label: "Meerut", state: "Uttar Pradesh" },
  { value: "moradabad", label: "Moradabad", state: "Uttar Pradesh" },
  { value: "muzaffarnagar", label: "Muzaffarnagar", state: "Uttar Pradesh" },
  { value: "varanasi", label: "Varanasi", state: "Uttar Pradesh" },
  // Madhya Pradesh
  { value: "bhopal", label: "Bhopal", state: "Madhya Pradesh" },
  { value: "gwalior", label: "Gwalior", state: "Madhya Pradesh" },
  { value: "indore", label: "Indore", state: "Madhya Pradesh" },
  { value: "jabalpur", label: "Jabalpur", state: "Madhya Pradesh" },
  { value: "rewa", label: "Rewa", state: "Madhya Pradesh" },
  { value: "sagar", label: "Sagar", state: "Madhya Pradesh" },
  { value: "ujjain", label: "Ujjain", state: "Madhya Pradesh" },
  // Rajasthan
  { value: "ajmer", label: "Ajmer", state: "Rajasthan" },
  { value: "alwar", label: "Alwar", state: "Rajasthan" },
  { value: "bikaner", label: "Bikaner", state: "Rajasthan" },
  { value: "jaipur", label: "Jaipur", state: "Rajasthan" },
  { value: "jodhpur", label: "Jodhpur", state: "Rajasthan" },
  { value: "kota", label: "Kota", state: "Rajasthan" },
  { value: "sikar", label: "Sikar", state: "Rajasthan" },
  { value: "udaipur", label: "Udaipur", state: "Rajasthan" },
  // Tamil Nadu
  { value: "chennai", label: "Chennai", state: "Tamil Nadu" },
  { value: "coimbatore", label: "Coimbatore", state: "Tamil Nadu" },
  { value: "erode", label: "Erode", state: "Tamil Nadu" },
  { value: "madurai", label: "Madurai", state: "Tamil Nadu" },
  { value: "salem", label: "Salem", state: "Tamil Nadu" },
  { value: "thanjavur", label: "Thanjavur", state: "Tamil Nadu" },
  { value: "tiruchirappalli", label: "Tiruchirappalli", state: "Tamil Nadu" },
  { value: "tirunelveli", label: "Tirunelveli", state: "Tamil Nadu" },
  // Andhra Pradesh / Telangana
  { value: "anantapur", label: "Anantapur", state: "Andhra Pradesh" },
  { value: "guntur", label: "Guntur", state: "Andhra Pradesh" },
  { value: "hyderabad", label: "Hyderabad", state: "Telangana" },
  { value: "karimnagar", label: "Karimnagar", state: "Telangana" },
  { value: "krishna", label: "Krishna", state: "Andhra Pradesh" },
  { value: "kurnool", label: "Kurnool", state: "Andhra Pradesh" },
  { value: "nalgonda", label: "Nalgonda", state: "Telangana" },
  { value: "warangal", label: "Warangal", state: "Telangana" },
  { value: "visakhapatnam", label: "Visakhapatnam", state: "Andhra Pradesh" },
  // West Bengal
  { value: "barddhaman", label: "Barddhaman", state: "West Bengal" },
  { value: "hooghly", label: "Hooghly", state: "West Bengal" },
  { value: "kolkata", label: "Kolkata", state: "West Bengal" },
  { value: "murshidabad", label: "Murshidabad", state: "West Bengal" },
  { value: "nadia", label: "Nadia", state: "West Bengal" },
  // Bihar
  { value: "bhagalpur", label: "Bhagalpur", state: "Bihar" },
  { value: "darbhanga", label: "Darbhanga", state: "Bihar" },
  { value: "gaya", label: "Gaya", state: "Bihar" },
  { value: "muzaffarpur", label: "Muzaffarpur", state: "Bihar" },
  { value: "patna", label: "Patna", state: "Bihar" },
  // Odisha
  { value: "balasore", label: "Balasore", state: "Odisha" },
  { value: "cuttack", label: "Cuttack", state: "Odisha" },
  { value: "ganjam", label: "Ganjam", state: "Odisha" },
  { value: "puri", label: "Puri", state: "Odisha" },
  // Kerala
  { value: "alappuzha", label: "Alappuzha", state: "Kerala" },
  { value: "ernakulam", label: "Ernakulam", state: "Kerala" },
  { value: "kozhikode", label: "Kozhikode", state: "Kerala" },
  { value: "palakkad", label: "Palakkad", state: "Kerala" },
  { value: "thrissur", label: "Thrissur", state: "Kerala" },
];

export default function Signup() {
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [district, setDistrict] = useState("");
  const [districtOpen, setDistrictOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  // Group districts by state for better organization
  const groupedDistricts = useMemo(() => {
    const groups: Record<string, typeof DISTRICTS> = {};
    DISTRICTS.forEach((d) => {
      if (!groups[d.state]) groups[d.state] = [];
      groups[d.state].push(d);
    });
    return groups;
  }, []);

  const selectedDistrict = DISTRICTS.find((d) => d.value === district);

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      // 1. Create auth user
      const { data: authData, error: authError } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            name,
            username,
            district: selectedDistrict?.label || district,
          },
        },
      });

      if (authError) throw authError;

      // 2. Create farmer profile in database
      if (authData.user) {
        const { error: profileError } = await supabase
          .from("farmers")
          .insert({
            user_id: authData.user.id,
            name,
            username,
            email,
            district: selectedDistrict?.label || district,
            state: selectedDistrict?.state || "",
          });

        if (profileError) {
          console.warn("Profile creation warning:", profileError);
          // Don't throw - auth was successful, profile can be created later
        }
      }

      toast({
        title: "Account created!",
        description: "Please check your email to verify your account.",
      });

      navigate("/login");
    } catch (error: any) {
      toast({
        title: "Signup failed",
        description: error.message || "Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo & Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
            <Leaf className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-3xl font-bold text-gradient-emerald font-display">
            Join AgroGuardian
          </h1>
          <p className="text-muted-foreground mt-2">
            Create your smart farming account
          </p>
        </div>

        {/* Signup Form */}
        <div className="dashboard-card p-8">
          <form onSubmit={handleSignup} className="space-y-5">
            {/* Full Name */}
            <div className="space-y-2">
              <Label htmlFor="name">Full Name</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="name"
                  type="text"
                  placeholder="Enter your full name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="pl-10"
                  required
                />
              </div>
            </div>

            {/* Username */}
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="username"
                  type="text"
                  placeholder="Choose a username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ""))}
                  className="pl-10"
                  required
                />
              </div>
            </div>

            {/* Email */}
            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  placeholder="farmer@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="pl-10"
                  required
                />
              </div>
            </div>

            {/* District Dropdown with Search */}
            <div className="space-y-2">
              <Label>District</Label>
              <Popover open={districtOpen} onOpenChange={setDistrictOpen}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={districtOpen}
                    className="w-full justify-between pl-10 relative"
                  >
                    <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                    {selectedDistrict
                      ? `${selectedDistrict.label}, ${selectedDistrict.state}`
                      : "Select your district..."}
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-full p-0" align="start">
                  <Command>
                    <CommandInput placeholder="Search district..." />
                    <CommandList className="max-h-[300px]">
                      <CommandEmpty>No district found.</CommandEmpty>
                      {Object.entries(groupedDistricts).map(([state, districts]) => (
                        <CommandGroup key={state} heading={state}>
                          {districts.map((d) => (
                            <CommandItem
                              key={d.value}
                              value={`${d.label} ${d.state}`}
                              onSelect={() => {
                                setDistrict(d.value);
                                setDistrictOpen(false);
                              }}
                            >
                              <Check
                                className={cn(
                                  "mr-2 h-4 w-4",
                                  district === d.value ? "opacity-100" : "opacity-0"
                                )}
                              />
                              {d.label}
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      ))}
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>

            {/* Password */}
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="password"
                  type="password"
                  placeholder="Create a strong password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10"
                  minLength={6}
                  required
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Minimum 6 characters
              </p>
            </div>

            <Button
              type="submit"
              className="w-full bg-primary hover:bg-primary/90"
              disabled={loading || !district}
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <UserPlus className="w-4 h-4 mr-2" />
                  Create Account
                </>
              )}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-muted-foreground">
              Already have an account?{" "}
              <Link
                to="/login"
                className="text-primary hover:underline font-medium"
              >
                Sign in here
              </Link>
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-muted-foreground mt-6">
          🌱 Green Growth Certified • Sustainable Farming Solutions
        </p>
      </div>
    </div>
  );
}
