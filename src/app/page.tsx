"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { motion, useScroll, useTransform, AnimatePresence } from "framer-motion";
import { ArrowRight, Code, ShieldCheck, Zap, Github, Linkedin, Layers, Search, CheckCircle, LogIn, LogOut, Clock, User } from "lucide-react";

// --- Framer Motion Standard Variants ---
const sectionVariant = {
  hidden: { opacity: 0, y: 32 },
  visible: { 
    opacity: 1, 
    y: 0, 
    transition: { 
      duration: 0.55, 
      ease: [0.22, 1, 0.36, 1] 
    } 
  }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariant = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

// --- Components ---

function Nav() {
  return (
    <nav className="fixed top-0 w-full z-50 px-6 py-4 flex justify-between items-center backdrop-blur-md border-b border-white/5 bg-slate-950/20">
      <div className="flex items-center gap-2">
        <Link href="/">
          <Image src="/logo.svg" alt="AuraDiff Logo" width={120} height={32} />
        </Link>
      </div>
      <div className="hidden md:flex gap-8 text-sm font-medium text-gray-400">
        <Link href="/compare" className="hover:text-indigo-400 transition-colors">Pairwise</Link>
        <Link href="/batch" className="hover:text-indigo-400 transition-colors">Batch Mode</Link>
        <a href="#algorithms" className="hover:text-indigo-400 transition-colors">Algorithms</a>
      </div>
      <div className="flex items-center gap-3">
        <Link 
          href="/compare" 
          className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-bold rounded-full transition-all shadow-lg shadow-indigo-500/20 flex items-center gap-2"
        >
          Get Started
        </Link>
      </div>
    </nav>
  );
}

function Hero() {
  const headline = "Detect Plagiarism. Instantly.";
  const characters = headline.split("");

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center pt-20 overflow-hidden bg-gradient-to-b from-slate-950 via-indigo-950 to-slate-900">
      {/* Floating Particles Placeholder (CSS Dots) */}
      <div className="absolute inset-0 opacity-15 pointer-events-none">
        <div className="absolute h-1 w-1 bg-white rounded-full top-[10%] left-[20%] animate-pulse" />
        <div className="absolute h-1 w-1 bg-white rounded-full top-[30%] left-[80%] animate-pulse delay-75" />
        <div className="absolute h-1.5 w-1.5 bg-indigo-400 rounded-full top-[60%] left-[15%] animate-bounce" />
        <div className="absolute h-1 w-1 bg-white rounded-full top-[85%] left-[70%] animate-pulse delay-150" />
      </div>

      <div className="container mx-auto px-6 grid md:grid-cols-2 gap-12 items-center relative z-10">
        <div className="space-y-8">
          <div className="flex flex-wrap text-5xl md:text-7xl font-black tracking-tighter">
            {characters.map((char, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05, duration: 0.3 }}
                className={char === 'P' || char === 'l' || char === 'a' || char === 'g' || char === 'i' || char === 'a' || char === 'r' || char === 'i' || char === 's' || char === 'm' || char === '.' ? 'text-indigo-400' : 'text-white'}
              >
                {char === " " ? "\u00A0" : char}
              </motion.span>
            ))}
          </div>

          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5, duration: 0.8 }}
            className="text-xl md:text-2xl text-gray-400 font-light"
          >
            Three AI algorithms. <span className="text-white font-medium">One truth.</span>
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8 }}
            className="flex gap-4"
          >
            <Link href="/compare" className="px-8 py-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-2xl transition-all shadow-xl shadow-indigo-600/30 flex items-center gap-2">
              Analyse Now <ArrowRight size={20} />
            </Link>
            <a href="#algorithms" className="px-8 py-4 border border-indigo-400/30 text-indigo-300 font-bold rounded-2xl hover:bg-white/5 transition-all">
              See It Live
            </a>
          </motion.div>
        </div>

        <motion.div 
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ type: "spring", damping: 20, stiffness: 100, delay: 1 }}
          className="relative hidden md:block"
        >
          <div className="glass-panel p-4 rotate-3 scale-110 shadow-2xl overflow-hidden border-white/10">
             <div className="w-full h-80 bg-slate-900 rounded-lg overflow-hidden relative">
                <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/10 to-transparent" />
                <div className="p-4 space-y-3">
                  <div className="h-2 w-2/3 bg-white/10 rounded-full" />
                  <div className="h-2 w-1/2 bg-white/10 rounded-full" />
                  <div className="h-2 w-3/4 bg-indigo-500/20 rounded-full" />
                  <div className="grid grid-cols-2 gap-4 mt-8">
                    <div className="aspect-video bg-white/5 rounded-lg border border-white/5" />
                    <div className="aspect-video bg-white/5 rounded-lg border border-white/5" />
                  </div>
                </div>
                <div className="absolute bottom-4 right-4 bg-indigo-600/20 border border-indigo-400/30 px-3 py-1.5 rounded-full text-[10px] font-bold text-indigo-300 uppercase tracking-widest">
                  Match Detected
                </div>
             </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function AlgorithmShowcase() {
  return (
    <section id="algorithms" className="py-32 bg-slate-950 overflow-hidden">
      <div className="container mx-auto px-6">
        <motion.div 
          variants={sectionVariant}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl font-bold mb-4">Multi-Algorithmic Core</h2>
          <p className="text-gray-500 max-w-xl mx-auto">AuraDiff combines three distinct detection engines to catch plagiarism no single algorithm can find alone.</p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Winnowing Card */}
          <motion.div 
            variants={sectionVariant}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="p-8 rounded-3xl bg-blue-400/5 border border-blue-400/20 hover:scale-105 transition-transform duration-500 group"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                <Search size={20} className="text-blue-400" />
              </div>
              <h3 className="text-2xl font-bold text-blue-400">Winnowing</h3>
            </div>
            <p className="text-gray-400 leading-relaxed mb-6">Fingerprints text using rolling k-gram hashes to detect exact copy-paste, even when surrounded by original content.</p>
            
            <div className="h-44 w-full bg-slate-950 rounded-2xl border border-white/5 p-4 overflow-hidden relative font-mono text-xs">
              <div className="space-y-1.5">
                <p className="text-gray-600">{"// Source document"}</p>
                <p><span className="text-gray-500">the quick </span><span className="bg-blue-500/30 text-blue-200 px-1 rounded">brown fox jumps over</span><span className="text-gray-500"> the lazy dog</span></p>
                <p className="text-gray-600 mt-3">{"// Target document"}</p>
                <p><span className="text-gray-500">a fast </span><span className="bg-blue-500/30 text-blue-200 px-1 rounded">brown fox jumps over</span><span className="text-gray-500"> the sleeping cat</span></p>
                <div className="mt-3 flex items-center gap-2 text-[10px]">
                  <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                  <span className="text-blue-400 font-bold">Hash match: k-gram #47 → 92% overlap</span>
                </div>
              </div>
              <div className="absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-slate-950 to-transparent" />
            </div>
          </motion.div>

          {/* Structural Card */}
          <motion.div 
            variants={sectionVariant}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            transition={{ delay: 0.1 }}
            className="p-8 rounded-3xl bg-emerald-400/5 border border-emerald-400/20 hover:scale-105 transition-transform duration-500 group"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                <Code size={20} className="text-emerald-400" />
              </div>
              <h3 className="text-2xl font-bold text-emerald-400">Structural</h3>
            </div>
            <p className="text-gray-400 leading-relaxed mb-6">Parses Abstract Syntax Trees to detect logic theft even when variable names, formatting, or comments are changed.</p>
            
            <div className="h-44 w-full bg-slate-950 rounded-2xl border border-white/5 p-4 overflow-hidden relative font-mono text-xs">
              <div className="space-y-1.5">
                <p className="text-gray-600">{"// Student A"}</p>
                <p><span className="text-emerald-400">def </span><span className="text-white">calc</span><span className="text-gray-500">(x, y):</span></p>
                <p className="ml-4"><span className="text-emerald-400">return</span> <span className="text-white">x * y + 1</span></p>
                <p className="text-gray-600 mt-2">{"// Student B (renamed)"}</p>
                <p><span className="text-emerald-400">def </span><span className="text-white">compute</span><span className="text-gray-500">(a, b):</span></p>
                <p className="ml-4"><span className="text-emerald-400">return</span> <span className="text-white">a * b + 1</span></p>
                <div className="mt-2 flex items-center gap-2 text-[10px]">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-emerald-400 font-bold">AST match: BinOp(Mult → Add) identical</span>
                </div>
              </div>
              <div className="absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-slate-950 to-transparent" />
            </div>
          </motion.div>

          {/* Semantic Card */}
          <motion.div 
            variants={sectionVariant}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            transition={{ delay: 0.2 }}
            className="p-8 rounded-3xl bg-violet-400/5 border border-violet-400/20 hover:scale-105 transition-transform duration-500 group"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center">
                <Zap size={20} className="text-violet-400" />
              </div>
              <h3 className="text-2xl font-bold text-violet-400">Semantic</h3>
            </div>
            <p className="text-gray-400 leading-relaxed mb-6">Uses sentence-transformer embeddings to catch AI-paraphrased content that preserves meaning but changes wording.</p>
            
            <div className="h-44 w-full bg-slate-950 rounded-2xl border border-white/5 p-4 overflow-hidden relative font-mono text-xs">
              <div className="space-y-1.5">
                <p className="text-gray-600">{"// Original"}</p>
                <p className="text-gray-300">{'"Machine learning models require training data"'}</p>
                <p className="text-gray-600 mt-2">{"// Paraphrased"}</p>
                <p className="text-gray-300">{'"AI systems need datasets for their learning process"'}</p>
                <div className="mt-3 space-y-1">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-gray-500">Cosine similarity</span>
                    <span className="text-violet-400 font-bold">0.89</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      whileInView={{ width: "89%" }}
                      viewport={{ once: true }}
                      transition={{ duration: 1.5, ease: "easeOut" }}
                      className="bg-violet-500 h-full rounded-full" 
                    />
                  </div>
                </div>
              </div>
              <div className="absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-slate-950 to-transparent" />
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function TrustBar() {
  const stats = [
    { label: "Algorithms", value: 3 },
    { label: "Accuracy", value: 99.2, suffix: "%" },
    { label: "Response", value: 2, prefix: "<", suffix: "s" },
    { label: "Files", value: "∞" }
  ];

  return (
    <section className="py-12 bg-slate-900 border-y border-indigo-800/30 overflow-hidden">
      <div className="container mx-auto px-6 flex flex-wrap justify-center gap-12 md:gap-24">
        {stats.map((s, i) => (
          <motion.div 
            key={i}
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="text-center"
          >
            <div className="text-4xl font-black text-white mb-1">
              {s.prefix}{s.value}{s.suffix}
            </div>
            <div className="text-xs font-bold text-gray-500 uppercase tracking-widest">{s.label}</div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

function Team() {
  const team = [
    { name: "P. Sandeep", role: "Lead Developer", initial: "PS" },
    { name: "Mahathi", role: "AI Engineer", initial: "M" },
    { name: "Lokesh", role: "Logic Architect", initial: "L" },
    { name: "Pramod", role: "UI/UX Designer", initial: "P" }
  ];

  return (
    <section className="py-32 container mx-auto px-6 overflow-hidden">
      <motion.div 
        variants={sectionVariant}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="text-center mb-20"
      >
        <h2 className="text-4xl font-bold mb-4">The Creators</h2>
        <p className="text-gray-500">Vignan Institute of Technology & Science · CSE Dept.</p>
      </motion.div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
        {team.map((member, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.1 }}
            className="group flex flex-col items-center text-center space-y-6"
          >
            <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-indigo-600 to-cyan-400 flex items-center justify-center text-3xl font-black group-hover:scale-110 transition-transform shadow-lg shadow-indigo-500/20">
              {member.initial}
            </div>
            <div>
              <h4 className="text-lg font-bold text-white mb-1">{member.name}</h4>
              <p className="text-xs text-indigo-400 font-bold uppercase tracking-wider">{member.role}</p>
            </div>
            <div className="flex gap-4 opacity-0 group-hover:opacity-100 transition-opacity">
              <Github size={18} className="text-gray-400 hover:text-white cursor-pointer" />
              <Linkedin size={18} className="text-gray-400 hover:text-indigo-400 cursor-pointer" />
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="py-20 border-t border-white/5 bg-slate-950/50">
      <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-12">
        <div className="space-y-4 max-w-xs text-center md:text-left">
          <Image src="/logo.svg" alt="AuraDiff Logo" width={140} height={40} />
          <p className="text-sm text-gray-500">Ensuring academic integrity and code originality through multi-algorithmic verification.</p>
        </div>
        <div className="flex flex-col items-center md:items-end gap-2">
          <p className="text-xs text-gray-600">© 2024–2025 Vignan Institute of Technology and Science</p>
          <p className="text-[10px] text-gray-700 uppercase tracking-widest font-bold">2nd Year B.Tech · CSE Department</p>
        </div>
      </div>
    </footer>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen selection:bg-indigo-500/30">
      <Nav />
      <Hero />
      <TrustBar />
      <AlgorithmShowcase />
      <Team />
      <Footer />
    </main>
  );
}
