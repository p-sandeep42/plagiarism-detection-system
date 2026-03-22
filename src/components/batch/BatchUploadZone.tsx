"use client";

import React, { useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, FileText, AlertCircle } from 'lucide-react';

interface BatchUploadZoneProps {
  onFilesAccepted: (files: File[]) => void;
  maxFiles: number;
}

export default function BatchUploadZone({ onFilesAccepted, maxFiles }: BatchUploadZoneProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, [files]);

  const handleFiles = (newFiles: File[]) => {
    const validExtensions = ['.py', '.js', '.txt', '.pdf', '.docx'];
    const filtered = newFiles.filter(f => 
      validExtensions.some(ext => f.name.toLowerCase().endsWith(ext))
    );

    if (filtered.length < newFiles.length) {
      setError("Some files were rejected. Only .py, .js, .txt, .pdf, .docx allowed.");
    } else {
      setError(null);
    }

    const updated = [...files, ...filtered].slice(0, maxFiles + 1);
    setFiles(updated);
    if (updated.length <= maxFiles) {
      onFilesAccepted(updated);
    }
  };

  const removeFile = (index: number) => {
    const updated = files.filter((_, i) => i !== index);
    setFiles(updated);
    onFilesAccepted(updated);
  };

  return (
    <div className="w-full space-y-4">
      <div 
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-xl p-12 transition-all flex flex-col items-center justify-center cursor-pointer
          ${files.length > maxFiles ? 'border-red-500/50 bg-red-500/5' : 'border-indigo-500/30 hover:border-indigo-500/50 bg-indigo-500/5'}`}
        onClick={() => {
          const input = document.createElement('input');
          input.type = 'file';
          input.multiple = true;
          input.onchange = (e) => handleFiles(Array.from((e.target as HTMLInputElement).files || []));
          input.click();
        }}
      >
        <Upload className="w-12 h-12 text-indigo-400 mb-4" />
        <p className="text-xl font-bold text-gray-200">Drop files here or click to browse</p>
        <p className="text-sm text-gray-500 mt-2">Up to {maxFiles} files (.py, .js, .txt, .pdf, .docx)</p>
      </div>

      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 text-amber-400 text-sm bg-amber-400/10 p-3 rounded-lg border border-amber-400/20">
          <AlertCircle className="w-4 h-4" /> {error}
        </motion.div>
      )}

      {files.length > maxFiles && (
        <div className="flex items-center gap-2 text-red-400 text-sm bg-red-400/10 p-3 rounded-lg border border-red-400/20">
          <AlertCircle className="w-4 h-4" /> Maximum {maxFiles} files allowed. Please remove {files.length - maxFiles} files.
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <AnimatePresence>
          {files.map((file, i) => (
            <motion.div 
              key={`${file.name}-${i}`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              layout
              className="flex items-center gap-2 bg-slate-800 border border-white/10 px-3 py-1.5 rounded-full text-sm text-gray-300"
            >
              <FileText className="w-4 h-4 text-indigo-400" />
              <span className="truncate max-w-[150px]">{file.name}</span>
              <button 
                onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                className="hover:text-red-400 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
