"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, LayoutDashboard, Split, Users, ExternalLink } from 'lucide-react';
import { BatchComparisonResponse, PairScore } from '@/types/backend';
import Dashboard from '../Dashboard';
import SplitView from '../SplitView';

interface StudentDetailPanelProps {
  studentIndex: number;
  data: BatchComparisonResponse;
  onClose: () => void;
}

export default function StudentDetailPanel({ studentIndex, data, onClose }: StudentDetailPanelProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'split' | 'all'>('overview');
  const student = data.students[studentIndex];
  const summary = data.summary.find(s => s.student_index === studentIndex);
  
  // Find worst peer
  let worstPeerIdx = -1;
  let maxScore = -1;
  data.matrix[studentIndex].forEach((pair, idx) => {
    if (pair && pair.total_score > maxScore) {
      maxScore = pair.total_score;
      worstPeerIdx = idx;
    }
  });

  const worstPair = worstPeerIdx !== -1 ? data.matrix[studentIndex][worstPeerIdx] : null;
  const worstPeer = worstPeerIdx !== -1 ? data.students[worstPeerIdx] : null;

  const tabs = [
    { id: 'overview', label: 'Score Overview', icon: LayoutDashboard },
    { id: 'split', label: 'Side-by-Side', icon: Split },
    { id: 'all', label: 'All Comparisons', icon: Users },
  ];

  return (
    <>
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-[100]"
      />
      <motion.div 
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed right-0 top-0 h-screen w-full max-w-4xl bg-slate-900 border-l border-white/10 z-[101] overflow-y-auto"
      >
        <div className="p-8 pb-32">
          {/* Header */}
          <div className="flex justify-between items-start mb-8">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-3xl font-black text-white">{student.name}</h2>
                <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest
                  ${summary?.risk_level === 'Critical' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 
                    summary?.risk_level === 'High' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                    summary?.risk_level === 'Medium' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                    'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}`}
                >
                  {summary?.risk_level} Risk
                </span>
              </div>
              <p className="text-gray-500 font-mono text-sm">{student.filename}</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors text-gray-400">
              <X size={24} />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-white/5 mb-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-6 py-4 text-sm font-medium transition-all relative
                  ${activeTab === tab.id ? 'text-indigo-400' : 'text-gray-500 hover:text-gray-300'}`}
              >
                <tab.icon size={18} />
                {tab.label}
                {activeTab === tab.id && (
                  <motion.div layoutId="activeTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-500" />
                )}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="space-y-8">
            {activeTab === 'overview' && (
              <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="bg-indigo-500/10 border border-indigo-500/20 p-6 rounded-2xl">
                  <p className="text-indigo-400 text-sm font-bold uppercase tracking-wider mb-2">Primary Match</p>
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-xl text-gray-200">Similarity found with <span className="font-bold text-white underline decoration-indigo-500/50">{worstPeer?.name}</span></p>
                    </div>
                    <div className="text-3xl font-black text-indigo-400">{(maxScore * 100).toFixed(1)}%</div>
                  </div>
                </div>
                <Dashboard data={{ scores: worstPair }} />
              </div>
            )}

            {activeTab === 'split' && (
              <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                <p className="text-gray-400 text-sm mb-4">Comparing <span className="text-indigo-400">{student.name}</span> against <span className="text-emerald-400">{worstPeer?.name}</span> (Worst Match)</p>
                <SplitView 
                  sourceText={data.parsed_texts[studentIndex]} 
                  targetText={data.parsed_texts[worstPeerIdx]} 
                  highlights={worstPair?.highlights || []} 
                />
              </div>
            )}

            {activeTab === 'all' && (
              <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <table className="w-full text-left">
                  <thead>
                    <tr className="text-xs font-bold text-gray-500 uppercase tracking-widest border-b border-white/5">
                      <th className="pb-4">Peer Student</th>
                      <th className="pb-4">Risk Level</th>
                      <th className="pb-4">Total Score</th>
                      <th className="pb-4">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.students.map((peer, idx) => {
                      if (idx === studentIndex) return null;
                      const pair = data.matrix[studentIndex][idx];
                      if (!pair) return null;
                      return (
                        <tr key={idx} className="border-b border-white/5 hover:bg-white/2 transition-colors group">
                          <td className="py-4 text-gray-300 font-medium">{peer.name}</td>
                          <td className="py-4">
                            <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase
                              ${pair.total_score > 0.8 ? 'bg-red-500/20 text-red-400' :
                                pair.total_score > 0.6 ? 'bg-orange-500/20 text-orange-400' :
                                pair.total_score > 0.3 ? 'bg-amber-500/20 text-amber-400' :
                                'bg-emerald-500/20 text-emerald-400'}`}
                            >
                              {pair.total_score > 0.8 ? 'Critical' : pair.total_score > 0.6 ? 'High' : pair.total_score > 0.3 ? 'Medium' : 'Low'}
                            </span>
                          </td>
                          <td className="py-4">
                            <div className="flex items-center gap-3">
                              <span className="text-sm font-bold text-gray-100 italic w-10">{(pair.total_score * 100).toFixed(0)}%</span>
                              <div className="w-24 bg-slate-800 h-1 rounded-full overflow-hidden">
                                <div 
                                  className="bg-indigo-500 h-full" 
                                  style={{ width: `${pair.total_score * 100}%` }}
                                />
                              </div>
                            </div>
                          </td>
                          <td className="py-4">
                            <button className="p-2 opacity-0 group-hover:opacity-100 transition-opacity text-indigo-400 hover:text-indigo-300">
                              <ExternalLink size={16} />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </>
  );
}
