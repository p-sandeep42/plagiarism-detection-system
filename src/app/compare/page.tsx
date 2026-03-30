"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import FileUpload from "@/components/FileUpload";
import Dashboard from "@/components/Dashboard";
import SplitView from "@/components/SplitView";
import { motion } from "framer-motion";
import { ArrowLeft, Layers, Loader2 } from "lucide-react";

export default function ComparePage() {
  const [analysisData, setAnalysisData] = useState<any>(null);
  const router = useRouter();

  return (
    <main className="min-h-screen py-20 px-4 md:px-8 max-w-7xl mx-auto flex flex-col items-center">
      <div className="w-full flex justify-between items-center mb-12">
        <Link href="/" className="text-gray-500 hover:text-white transition flex items-center gap-2 text-sm font-medium">
          <ArrowLeft size={16} /> Back to Home
        </Link>
        <Link href="/batch" className="flex items-center gap-2 px-4 py-2 bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 text-sm font-bold rounded-full hover:bg-indigo-600/30 transition-all">
          <Layers size={16} /> Batch Mode
        </Link>
      </div>

      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-16"
      >
        <div className="flex justify-center mb-4">
          <Image src="/favicon.png" alt="Plagiarism Detection" width={56} height={56} className="rounded-xl" />
        </div>
        <h1 className="text-5xl md:text-7xl font-black mb-4 tracking-tight">
          Pairwise <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-600">Comparison</span>
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto font-light">
          Context-aware document comparison. Detect direct plagiarism, structural mirroring, and AI paraphrasing.
        </p>
      </motion.div>

      {!analysisData ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="w-full"
        >
          <FileUpload onUploadComplete={(data) => setAnalysisData(data)} />
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="w-full space-y-12"
        >
          <button
            onClick={() => setAnalysisData(null)}
            className="text-gray-400 hover:text-white transition flex items-center mb-4 text-sm font-semibold tracking-wide uppercase"
          >
            ← Upload New Files
          </button>

          <Dashboard data={analysisData} />

          {analysisData.highlights && (
            <SplitView
              sourceText={analysisData.source_text}
              targetText={analysisData.target_text}
              highlights={analysisData.highlights}
            />
          )}
        </motion.div>
      )}
    </main>
  );
}
