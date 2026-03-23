"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, Clock, FileText, Users, AlertTriangle, ChevronRight, Loader2, Trash2 } from "lucide-react";
import { useAuth } from "@/lib/AuthContext";
import ErrorScreen from "@/components/ErrorScreen";

interface HistoryEntry {
  id: number;
  comparison_type: string;
  filenames: string[];
  total_score: number;
  risk_level: string;
  created_at: string;
}

const riskColors: Record<string, string> = {
  Low: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
  Medium: "text-amber-400 bg-amber-500/10 border-amber-500/20",
  High: "text-orange-400 bg-orange-500/10 border-orange-500/20",
  Critical: "text-red-400 bg-red-500/10 border-red-500/20",
};

export default function HistoryPage() {
  const { isAuthenticated, isLoading: authLoading, token } = useAuth();
  const router = useRouter();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (authLoading) return;
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    fetch("/api/py/history", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load history");
        return r.json();
      })
      .then(setEntries)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [isAuthenticated, authLoading, token, router]);

  if (authLoading || (!isAuthenticated && !authLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="animate-spin text-indigo-400" size={32} />
      </div>
    );
  }

  if (error) {
    return (
      <main className="min-h-screen bg-slate-950 text-white p-8">
        <ErrorScreen type="500" message={error} />
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white p-8 md:p-12">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <Link href="/compare" className="text-gray-500 hover:text-white transition flex items-center gap-2 text-sm font-medium">
            <ArrowLeft size={16} /> Back
          </Link>
        </div>

        <div className="space-y-2">
          <h1 className="text-4xl font-black tracking-tight flex items-center gap-3">
            <Clock className="text-indigo-400" size={32} />
            Analysis History
          </h1>
          <p className="text-gray-400">View your past plagiarism analysis results.</p>
        </div>

        {/* Entries */}
        {loading ? (
          <div className="flex justify-center py-20">
            <Loader2 className="animate-spin text-indigo-400" size={32} />
          </div>
        ) : entries.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20 space-y-4"
          >
            <FileText size={48} className="text-gray-600 mx-auto" />
            <h3 className="text-xl font-bold text-gray-400">No History Yet</h3>
            <p className="text-gray-500">Run your first comparison to see results here.</p>
            <Link
              href="/compare"
              className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl transition-all"
            >
              Start Comparing
            </Link>
          </motion.div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence>
              {entries.map((entry, i) => (
                <motion.div
                  key={entry.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="group p-5 rounded-2xl border border-white/5 bg-slate-900/40 hover:bg-slate-800/50 transition-all cursor-default"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1 min-w-0">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${entry.comparison_type === "batch" ? "bg-violet-500/15" : "bg-blue-500/15"}`}>
                        {entry.comparison_type === "batch" ? (
                          <Users size={18} className="text-violet-400" />
                        ) : (
                          <FileText size={18} className="text-blue-400" />
                        )}
                      </div>

                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold uppercase tracking-wider text-gray-500">
                            {entry.comparison_type === "batch" ? "Batch" : "Pairwise"}
                          </span>
                          <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${riskColors[entry.risk_level] || riskColors.Low}`}>
                            {entry.risk_level}
                          </span>
                        </div>
                        <p className="text-sm text-gray-300 truncate mt-1">
                          {entry.filenames.join(", ")}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-6 flex-shrink-0 ml-4">
                      <div className="text-right">
                        <p className="text-lg font-black text-white">{Math.round(entry.total_score * 100)}%</p>
                        <p className="text-xs text-gray-500">
                          {new Date(entry.created_at).toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </main>
  );
}
