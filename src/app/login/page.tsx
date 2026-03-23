"use client";

import React, { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Mail, Lock, ArrowRight, Eye, EyeOff, Loader2, ShieldCheck } from "lucide-react";
import { useAuth } from "@/lib/AuthContext";

export default function LoginPage() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const { login, register } = useAuth();

  // Basic sanitization
  const sanitize = (val: string) => val.replace(/[<>]/g, "").trim();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const cleanEmail = sanitize(email);
    const cleanPassword = password; // Don't strip chars from password

    if (!cleanEmail || !cleanPassword) {
      setError("Please fill in all fields.");
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (cleanPassword.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    setLoading(true);
    try {
      if (mode === "login") {
        await login(cleanEmail, cleanPassword);
      } else {
        await register(cleanEmail, cleanPassword);
      }
      router.push("/compare");
    } catch (err: any) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center px-4 py-12 bg-gradient-to-br from-slate-950 via-indigo-950/30 to-slate-950">
      {/* Background glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] rounded-full bg-indigo-600/8 blur-[120px]" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative w-full max-w-md"
      >
        {/* Logo */}
        <Link href="/" className="flex justify-center mb-8">
          <Image src="/logo.svg" alt="AuraDiff" width={140} height={40} />
        </Link>

        {/* Card */}
        <div className="rounded-3xl border border-white/8 bg-slate-900/60 backdrop-blur-xl shadow-2xl shadow-indigo-500/5 p-8 space-y-6">
          {/* Header */}
          <div className="text-center space-y-2">
            <div className="w-12 h-12 rounded-xl bg-indigo-600/20 border border-indigo-500/20 flex items-center justify-center mx-auto mb-4">
              <ShieldCheck size={24} className="text-indigo-400" />
            </div>
            <h1 className="text-2xl font-black text-white">
              {mode === "login" ? "Welcome Back" : "Create Account"}
            </h1>
            <p className="text-sm text-gray-400">
              {mode === "login"
                ? "Sign in to access your analysis history"
                : "Get started with plagiarism detection"}
            </p>
          </div>

          {/* Toggle */}
          <div className="flex rounded-xl bg-slate-800/50 p-1">
            <button
              onClick={() => { setMode("login"); setError(null); }}
              className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all ${mode === "login" ? "bg-indigo-600 text-white shadow-md" : "text-gray-400 hover:text-white"}`}
            >
              Login
            </button>
            <button
              onClick={() => { setMode("register"); setError(null); }}
              className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all ${mode === "register" ? "bg-indigo-600 text-white shadow-md" : "text-gray-400 hover:text-white"}`}
            >
              Register
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1">
              <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Email</label>
              <div className="relative">
                <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                  id="login-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  autoComplete="email"
                  className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-white/8 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Password</label>
              <div className="relative">
                <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                  id="login-password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  className="w-full pl-10 pr-12 py-3 bg-slate-800/50 border border-white/8 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition"
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              {mode === "register" && (
                <p className="text-xs text-gray-500 mt-1">Minimum 6 characters</p>
              )}
            </div>

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                className="px-4 py-2.5 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm font-medium"
              >
                {error}
              </motion.div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all shadow-lg shadow-indigo-500/20"
            >
              {loading ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <>
                  {mode === "login" ? "Sign In" : "Create Account"}
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-gray-600 mt-6">
          By continuing, you agree to AuraDiff&apos;s Terms of Service.
        </p>
      </motion.div>
    </main>
  );
}
