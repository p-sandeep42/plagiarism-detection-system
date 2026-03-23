"use client";

import React, { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers, Play, AlertCircle, CheckCircle2, Loader2, Info, ArrowLeft, Home } from 'lucide-react';
import { getBackend } from '@/lib/backend';
import { BatchComparisonResponse } from '@/types/backend';
import BatchUploadZone from '@/components/batch/BatchUploadZone';
import BatchErrorBoundary from '@/components/batch/BatchErrorBoundary';
import StudentMatrix from '@/components/batch/StudentMatrix';
import StudentDetailPanel from '@/components/batch/StudentDetailPanel';
import { useAuth } from '@/lib/AuthContext';
import { useToast } from '@/components/Toast';
import ErrorScreen from '@/components/ErrorScreen';

export default function BatchPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchComparisonResponse | null>(null);
  const [selectedStudent, setSelectedStudent] = useState<number | null>(null);
  const [progress, setProgress] = useState<{ completed: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const { addToast } = useToast();
  const router = useRouter();
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [authLoading, isAuthenticated, router]);

  useEffect(() => {
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  const startAnalysis = async () => {
    if (files.length < 2) return;
    
    setLoading(true);
    setResult(null);
    setError(null);
    setProgress({ completed: 0, total: (files.length * (files.length - 1)) / 2 });

    const sessionId = Math.random().toString(36).substring(2);
    const token = process.env.NEXT_PUBLIC_BATCH_WS_SECRET || 'replace_with_random_32_char_string';

    // Connect WebSocket for progress
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host.includes('localhost') ? 'localhost:8000' : window.location.host}/ws/batch-progress?token=${token}`;
    
    try {
      ws.current = new WebSocket(wsUrl);
      ws.current.onopen = () => {
        ws.current?.send(JSON.stringify({ session_id: sessionId }));
      };
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setProgress({ completed: data.completed, total: data.total });
      };
      ws.current.onerror = () => {
        console.warn("WebSocket connection failed - progress updates unavailable");
      };
    } catch (e) {
      console.warn("WS connection failed", e);
    }

    try {
      const backend = await getBackend();
      const response = await backend.compareBatch(files, sessionId);
      setResult(response);
      addToast("success", `Batch analysis complete! ${response.students.length} documents scanned.`);
    } catch (err: any) {
      const msg = err.message || "An error occurred during batch analysis.";
      setError(msg);
      addToast("error", msg);
    } finally {
      setLoading(false);
      if (ws.current) ws.current.close();
    }
  };

  if (authLoading) {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <Loader2 className="animate-spin text-indigo-400" size={32} />
      </main>
    );
  }

  if (!isAuthenticated) return null;

  return (
    <main className="min-h-screen bg-slate-950 text-white p-8 md:p-12 selection:bg-indigo-500/30">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Navigation Bar */}
        <div className="flex justify-between items-center">
          <Link href="/" className="text-gray-500 hover:text-white transition flex items-center gap-2 text-sm font-medium">
            <ArrowLeft size={16} /> Back to Home
          </Link>
          <Link href="/compare" className="flex items-center gap-2 px-4 py-2 bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 text-sm font-bold rounded-full hover:bg-indigo-600/30 transition-all">
            Pairwise Compare
          </Link>
        </div>

        {/* Header */}
        <section className="space-y-4">
          <div className="flex items-center gap-4">
            <Image src="/favicon.png" alt="Plagiarism Detection" width={48} height={48} className="rounded-xl" />
            <div>
              <h1 className="text-4xl font-black tracking-tight">Batch Analysis</h1>
              <p className="text-gray-400 max-w-2xl leading-relaxed mt-1">
                Upload multiple student submissions to identify cross-plagiarism and structural similarities across an entire class group.
              </p>
            </div>
          </div>
        </section>

        {/* Upload Zone */}
        {!result && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            <BatchUploadZone onFilesAccepted={setFiles} maxFiles={30} />
            
            <div className="flex justify-center">
              <button
                onClick={startAnalysis}
                disabled={files.length < 2 || loading}
                className={`flex items-center gap-3 px-10 py-5 rounded-2xl text-lg font-bold transition-all
                  ${files.length < 2 || loading 
                    ? 'bg-slate-800 text-gray-500 cursor-not-allowed border border-white/5' 
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-2xl shadow-indigo-500/20 scale-100 hover:scale-105 active:scale-95'}`}
              >
                {loading ? (
                  <><Loader2 className="animate-spin" /> Analyzing Group...</>
                ) : (
                  <><Play fill="currentColor" /> Run Batch Comparison</>
                )}
              </button>
            </div>
          </motion.div>
        )}

        {/* Loading / Progress State */}
        {loading && progress && (
          <div className="max-w-xl mx-auto text-center space-y-6 py-12">
            <div className="relative w-32 h-32 mx-auto">
              <svg className="w-full h-full transform -rotate-90">
                <circle cx="64" cy="64" r="60" stroke="rgba(255,255,255,0.05)" strokeWidth="8" fill="none" />
                <motion.circle 
                  cx="64" cy="64" r="60" 
                  stroke="#4f46e5" 
                  strokeWidth="8" 
                  fill="none"
                  strokeDasharray="377"
                  animate={{ strokeDashoffset: 377 - (377 * (progress.completed / progress.total)) }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center font-black text-2xl">
                {Math.round((progress.completed / progress.total) * 100)}%
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-bold">Cross-checking Pairs</h3>
              <p className="text-gray-500 text-sm">Processing {progress.completed} of {progress.total} comparisons...</p>
            </div>
          </div>
        )}

        {/* Results State */}
        <AnimatePresence>
          {result && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-12"
            >
              <div className="flex items-center gap-4 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl">
                <div className="w-10 h-10 bg-emerald-500/20 rounded-full flex items-center justify-center text-emerald-400">
                  <CheckCircle2 size={24} />
                </div>
                <div>
                  <h4 className="font-bold text-gray-200">Analysis Complete</h4>
                  <p className="text-sm text-gray-500">Successfully scanned {result.students.length} documents and mapped all similarity indices.</p>
                </div>
                <button 
                  onClick={() => { setResult(null); setFiles([]); }}
                  className="ml-auto text-sm text-gray-400 hover:text-white underline underline-offset-4"
                >
                  Start New Batch
                </button>
              </div>

              <BatchErrorBoundary>
                <StudentMatrix data={result} onStudentClick={setSelectedStudent} />
              </BatchErrorBoundary>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Detail Sidebar */}
        <AnimatePresence>
          {selectedStudent !== null && result && (
            <StudentDetailPanel 
              studentIndex={selectedStudent} 
              data={result} 
              onClose={() => setSelectedStudent(null)} 
            />
          )}
        </AnimatePresence>

        {/* Error State */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-6 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center gap-4 text-red-400"
          >
            <AlertCircle />
            <div>
              <p className="font-bold">Analysis Failed</p>
              <p className="text-sm opacity-80">{error}</p>
            </div>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-sm underline underline-offset-4 hover:text-red-300"
            >
              Dismiss
            </button>
          </motion.div>
        )}

        {/* Info Footer */}
        <p className="text-center text-gray-600 text-xs flex items-center justify-center gap-2 pt-12">
          <Info size={14} /> Matrix computation time varies with student count O(n²).
        </p>
      </div>
    </main>
  );
}
