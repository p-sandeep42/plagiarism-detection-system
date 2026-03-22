"use client";

import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Tooltip } from "recharts";
import { motion } from "framer-motion";

export default function Dashboard({ data }: { data: any }) {
  const scores = data?.scores || {
    winnowing_score: 0.1,
    jaccard_score: 0.2,
    semantic_score: 0.8,
    total_score: 0.45,
  };

  const chartData = [
    { subject: 'Exact (Winnowing)', A: scores.winnowing_score * 100, fullMark: 100 },
    { subject: 'Structural (Jaccard)', A: scores.jaccard_score * 100, fullMark: 100 },
    { subject: 'Contextual (Semantic)', A: scores.semantic_score * 100, fullMark: 100 },
  ];

  const totalPercentage = (scores.total_score * 100).toFixed(1);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-5xl mx-auto mt-12 grid grid-cols-1 md:grid-cols-2 gap-8"
    >
      <div className="glass-panel p-8 flex flex-col items-center justify-center">
        <h3 className="text-xl font-bold mb-4 text-gray-300">Total Plagiarism Risk</h3>
        <div className="relative flex items-center justify-center w-48 h-48">
          <svg className="w-full h-full transform -rotate-90">
            <circle cx="96" cy="96" r="80" stroke="rgba(255,255,255,0.1)" strokeWidth="12" fill="none" />
            <motion.circle 
              initial={{ strokeDasharray: "0 500" }}
              animate={{ strokeDasharray: `${Number(totalPercentage) * 5.02} 500` }}
              transition={{ duration: 1.5, ease: "easeOut" }}
              cx="96" cy="96" r="80" 
              stroke={Number(totalPercentage) > 50 ? "#ef4444" : "#3b82f6"} 
              strokeWidth="12" 
              fill="none" 
              strokeLinecap="round" 
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-black">{totalPercentage}%</span>
            <span className="text-xs text-gray-500 mt-1 uppercase tracking-widest">Similarity</span>
          </div>
        </div>
      </div>

      <div className="glass-panel p-8 flex flex-col md:flex-row gap-8 items-center w-full">
        <div className="w-full md:w-1/2 flex flex-col">
          <h3 className="text-xl font-bold mb-4 text-gray-300">Algorithmic Breakdown</h3>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name="Score" dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(10,10,11,0.9)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="w-full md:w-1/2 flex flex-col justify-center space-y-6 bg-black/30 p-6 rounded-xl border border-white/5">
            <div>
                <div className="flex justify-between items-center mb-1"><span className="text-gray-400 font-medium">Exact Match (Winnowing)</span><span className="font-bold text-blue-400">{(scores.winnowing_score * 100).toFixed(1)}%</span></div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                    <div className="bg-blue-400 h-2 rounded-full shadow-[0_0_10px_rgba(59,130,246,0.5)]" style={{ width: `${scores.winnowing_score * 100}%` }}></div>
                </div>
            </div>
            <div>
                <div className="flex justify-between items-center mb-1"><span className="text-gray-400 font-medium">Structural (Jaccard / AST)</span><span className="font-bold text-emerald-400">{(scores.jaccard_score * 100).toFixed(1)}%</span></div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                    <div className="bg-emerald-400 h-2 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]" style={{ width: `${scores.jaccard_score * 100}%` }}></div>
                </div>
            </div>
            <div>
                <div className="flex justify-between items-center mb-1"><span className="text-gray-400 font-medium">Contextual (Semantic)</span><span className="font-bold text-purple-400">{(scores.semantic_score * 100).toFixed(1)}%</span></div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                    <div className="bg-purple-400 h-2 rounded-full shadow-[0_0_10px_rgba(168,85,247,0.5)]" style={{ width: `${scores.semantic_score * 100}%` }}></div>
                </div>
            </div>
        </div>
      </div>
    </motion.div>
  );
}
