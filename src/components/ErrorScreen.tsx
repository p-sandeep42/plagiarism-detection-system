"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ShieldX, Lock, FileQuestion, ServerCrash, WifiOff, Home, LogIn, RotateCcw } from "lucide-react";

interface ErrorScreenProps {
  type: "401" | "403" | "404" | "500" | "network";
  message?: string;
}

const errorConfig = {
  "401": {
    icon: Lock,
    title: "Authentication Required",
    description: "You need to log in to access this page.",
    color: "amber",
    action: { label: "Login", href: "/login", icon: LogIn },
  },
  "403": {
    icon: ShieldX,
    title: "Access Denied",
    description: "You don't have permission to access this resource.",
    color: "red",
    action: { label: "Go Home", href: "/", icon: Home },
  },
  "404": {
    icon: FileQuestion,
    title: "Page Not Found",
    description: "The page you're looking for doesn't exist or has been moved.",
    color: "blue",
    action: { label: "Go Home", href: "/", icon: Home },
  },
  "500": {
    icon: ServerCrash,
    title: "Server Error",
    description: "Something went wrong on our end. Please try again later.",
    color: "red",
    action: null,
  },
  network: {
    icon: WifiOff,
    title: "Connection Failed",
    description: "Unable to reach the server. Check your internet connection.",
    color: "amber",
    action: null,
  },
};

const colorClasses: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  red: {
    bg: "bg-red-500/10",
    border: "border-red-500/20",
    text: "text-red-400",
    glow: "shadow-red-500/10",
  },
  amber: {
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
    text: "text-amber-400",
    glow: "shadow-amber-500/10",
  },
  blue: {
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
    text: "text-blue-400",
    glow: "shadow-blue-500/10",
  },
};

export default function ErrorScreen({ type, message }: ErrorScreenProps) {
  const config = errorConfig[type];
  const colors = colorClasses[config.color];
  const Icon = config.icon;

  return (
    <div className="min-h-[60vh] flex items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="text-center max-w-md space-y-6"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.1, type: "spring", stiffness: 200 }}
          className={`w-20 h-20 rounded-2xl ${colors.bg} ${colors.border} border flex items-center justify-center mx-auto shadow-xl ${colors.glow}`}
        >
          <Icon size={36} className={colors.text} />
        </motion.div>

        <div className="space-y-2">
          <h2 className="text-3xl font-black text-white">{config.title}</h2>
          <p className="text-gray-400 leading-relaxed">
            {message || config.description}
          </p>
        </div>

        <div className="flex items-center justify-center gap-4 pt-4">
          {config.action && (
            <Link
              href={config.action.href}
              className="flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-indigo-500/20"
            >
              <config.action.icon size={16} />
              {config.action.label}
            </Link>
          )}
          <button
            onClick={() => window.location.reload()}
            className="flex items-center gap-2 px-6 py-3 border border-white/10 text-gray-400 hover:text-white font-medium rounded-xl transition-all hover:bg-white/5"
          >
            <RotateCcw size={16} />
            Try Again
          </button>
        </div>
      </motion.div>
    </div>
  );
}
