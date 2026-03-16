"use client";

import React, { useState } from 'react';
import { BatchComparisonResponse, PairScore } from '@/types/backend';
import { Download, ChevronDown, ChevronRight, FileText, AlertTriangle, ShieldCheck, ShieldAlert, ShieldX, Search, Code, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface StudentMatrixProps {
  data: BatchComparisonResponse;
  onStudentClick: (index: number) => void;
}

function getRiskBadge(level: string) {
  switch (level) {
    case 'Critical': return { bg: 'bg-red-500/15 border-red-500/30', text: 'text-red-400', icon: <ShieldX size={14} /> };
    case 'High':     return { bg: 'bg-orange-500/15 border-orange-500/30', text: 'text-orange-400', icon: <ShieldAlert size={14} /> };
    case 'Medium':   return { bg: 'bg-amber-500/15 border-amber-500/30', text: 'text-amber-400', icon: <AlertTriangle size={14} /> };
    default:         return { bg: 'bg-emerald-500/15 border-emerald-500/30', text: 'text-emerald-400', icon: <ShieldCheck size={14} /> };
  }
}

function ScoreBar({ label, value, color, icon }: { label: string; value: number; color: string; icon: React.ReactNode }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1.5 w-28 shrink-0">
        <span className={`${color}`}>{icon}</span>
        <span className="text-xs text-gray-400 font-medium">{label}</span>
      </div>
      <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className={`h-full rounded-full ${color.replace('text-', 'bg-')}`}
        />
      </div>
      <span className={`text-sm font-bold w-12 text-right ${color}`}>{pct}%</span>
    </div>
  );
}

function PeerComparison({ pair, peerName, peerFilename }: { pair: PairScore; peerName: string; peerFilename: string }) {
  const totalPct = Math.round(pair.total_score * 100);
  const riskColor = pair.total_score > 0.8 ? 'text-red-400' : pair.total_score > 0.6 ? 'text-orange-400' : pair.total_score > 0.3 ? 'text-amber-400' : 'text-emerald-400';
  const riskBg = pair.total_score > 0.8 ? 'bg-red-500/10 border-red-500/20' : pair.total_score > 0.6 ? 'bg-orange-500/10 border-orange-500/20' : pair.total_score > 0.3 ? 'bg-amber-500/10 border-amber-500/20' : 'bg-emerald-500/10 border-emerald-500/20';

  return (
    <div className={`p-5 rounded-xl border ${riskBg} space-y-4`}>
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center">
            <FileText size={16} className="text-gray-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-gray-200">{peerName}</p>
            <p className="text-[11px] text-gray-500 font-mono">{peerFilename}</p>
          </div>
        </div>
        <div className="text-right">
          <p className={`text-2xl font-black ${riskColor}`}>{totalPct}%</p>
          <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Total</p>
        </div>
      </div>

      <div className="space-y-2.5">
        <ScoreBar label="Winnowing" value={pair.winnowing_score} color="text-blue-400" icon={<Search size={12} />} />
        <ScoreBar label="Structural" value={pair.jaccard_score} color="text-emerald-400" icon={<Code size={12} />} />
        <ScoreBar label="Semantic" value={pair.semantic_score} color="text-violet-400" icon={<Zap size={12} />} />
      </div>
    </div>
  );
}

export default function StudentMatrix({ data, onStudentClick }: StudentMatrixProps) {
  const [expandedStudent, setExpandedStudent] = useState<number | null>(null);
  const [sortBy, setSortBy] = useState<'index' | 'max_score'>('max_score');

  const sortedStudents = [...data.summary].sort((a, b) => {
    if (sortBy === 'index') return a.student_index - b.student_index;
    return b.max_score - a.max_score;
  });

  const toggleExpand = (idx: number) => {
    setExpandedStudent(expandedStudent === idx ? null : idx);
  };

  const downloadCSV = () => {
    let csv = 'Student,' + data.students.map(s => s.name).join(',') + '\n';
    data.students.forEach((s, i) => {
      let row = s.name + ',';
      row += data.matrix[i].map(cell => cell ? cell.total_score.toFixed(4) : '--').join(',');
      csv += row + '\n';
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'auradiff_results.csv';
    a.click();
  };

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-200">Submitted Documents</h2>
          <p className="text-sm text-gray-500 mt-1">{data.students.length} files analyzed · Click any document to view detailed scores</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={() => setSortBy(sortBy === 'index' ? 'max_score' : 'index')}
            className="flex items-center gap-2 px-4 py-2.5 bg-slate-800 border border-white/5 rounded-xl text-sm text-gray-300 hover:bg-slate-700 transition-colors"
          >
            {sortBy === 'index' ? 'Sort by Risk ↓' : 'Sort by Name ↓'}
          </button>
          <button 
            onClick={downloadCSV}
            className="flex items-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-medium transition-all shadow-lg shadow-indigo-500/20"
          >
            <Download size={16} /> Export CSV
          </button>
        </div>
      </div>

      {/* Document Cards */}
      <div className="space-y-3">
        {sortedStudents.map((summary) => {
          const i = summary.student_index;
          const student = data.students[i];
          const isExpanded = expandedStudent === i;
          const badge = getRiskBadge(summary.risk_level);
          const maxPct = Math.round(summary.max_score * 100);
          const avgPct = Math.round(summary.avg_score * 100);

          // Get peers sorted by score (highest first)
          const peers = data.students
            .map((peer, idx) => ({ peer, idx, pair: data.matrix[i][idx] }))
            .filter(p => p.idx !== i && p.pair !== null)
            .sort((a, b) => (b.pair?.total_score ?? 0) - (a.pair?.total_score ?? 0));

          return (
            <motion.div 
              key={i}
              layout
              className="rounded-2xl border border-white/5 bg-slate-900/50 overflow-hidden"
            >
              {/* Card Header — always visible */}
              <button
                onClick={() => toggleExpand(i)}
                className="w-full flex items-center gap-4 p-5 hover:bg-white/[0.02] transition-colors text-left"
              >
                {/* File icon with index */}
                <div className="w-12 h-12 rounded-xl bg-slate-800 border border-white/5 flex flex-col items-center justify-center shrink-0">
                  <FileText size={18} className="text-indigo-400" />
                  <span className="text-[9px] text-gray-500 font-bold mt-0.5">#{i + 1}</span>
                </div>

                {/* Name & filename */}
                <div className="flex-1 min-w-0">
                  <p className="text-base font-bold text-gray-200 truncate">{student.name}</p>
                  <p className="text-xs text-gray-500 font-mono truncate">{student.filename}</p>
                </div>

                {/* Quick stats */}
                <div className="hidden sm:flex items-center gap-6 shrink-0">
                  <div className="text-center">
                    <p className="text-xs text-gray-500 font-medium">Highest Match</p>
                    <p className={`text-lg font-black ${badge.text}`}>{maxPct}%</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-500 font-medium">Average</p>
                    <p className="text-lg font-black text-gray-300">{avgPct}%</p>
                  </div>
                </div>

                {/* Risk badge */}
                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-bold uppercase tracking-wider shrink-0 ${badge.bg} ${badge.text}`}>
                  {badge.icon}
                  {summary.risk_level}
                </div>

                {/* Expand chevron */}
                <div className="shrink-0 text-gray-500">
                  {isExpanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
                </div>
              </button>

              {/* Expanded Details */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: "easeInOut" }}
                    className="overflow-hidden"
                  >
                    <div className="px-5 pb-6 pt-2 border-t border-white/5">
                      {/* Section header */}
                      <div className="flex justify-between items-center mb-4">
                        <p className="text-sm text-gray-400">
                          Compared against <span className="text-white font-semibold">{peers.length} other documents</span>
                        </p>
                        <button
                          onClick={(e) => { e.stopPropagation(); onStudentClick(i); }}
                          className="text-xs text-indigo-400 hover:text-indigo-300 font-bold uppercase tracking-wider transition-colors"
                        >
                          Open Full Detail →
                        </button>
                      </div>

                      {/* Peer comparison cards */}
                      <div className="grid gap-3 md:grid-cols-2">
                        {peers.map(({ peer, idx, pair }) => (
                          <PeerComparison 
                            key={idx} 
                            pair={pair!} 
                            peerName={peer.name}
                            peerFilename={peer.filename}
                          />
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 pt-4 border-t border-white/5">
        <span className="text-xs text-gray-600 font-bold uppercase tracking-widest mr-2">Risk Levels:</span>
        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-emerald-500/40"></div><span className="text-xs text-gray-500">Low (&lt;30%)</span></div>
        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-amber-500/40"></div><span className="text-xs text-gray-500">Medium (30-60%)</span></div>
        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-orange-500/40"></div><span className="text-xs text-gray-500">High (60-80%)</span></div>
        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-red-500/40"></div><span className="text-xs text-gray-500">Critical (&gt;80%)</span></div>
      </div>
    </div>
  );
}
