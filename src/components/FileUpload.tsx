"use client";

import { useState } from "react";
import { UploadCloud, File, AlertCircle } from "lucide-react";
import { useAuth } from "@/lib/AuthContext";
import { useToast } from "@/components/Toast";

const ALLOWED_EXTENSIONS = ["txt", "py", "js", "ts", "java", "cpp", "c", "h", "md", "csv", "json", "pdf", "docx"];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

function sanitizeFilename(name: string): string {
  return name.replace(/[<>&"']/g, "").replace(/\.\./g, "").slice(0, 255);
}

function validateFile(file: File): string | null {
  const ext = file.name.split(".").pop()?.toLowerCase() || "";
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `File type '.${ext}' is not allowed.`;
  }
  if (file.size > MAX_FILE_SIZE) {
    return `File '${file.name}' exceeds 10MB limit.`;
  }
  if (file.size === 0) {
    return `File '${file.name}' is empty.`;
  }
  return null;
}

export default function FileUpload({ onUploadComplete }: { onUploadComplete: (data: any) => void }) {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { token } = useAuth();
  const { addToast } = useToast();

  const handleFileSelect = (file: File, setter: (f: File) => void) => {
    const err = validateFile(file);
    if (err) {
      setError(err);
      addToast("error", err);
      return;
    }
    setter(file);
    setError(null);
  };

  const handleCompare = async () => {
    if (!file1 || !file2) {
      setError("Please select two files to compare.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file1", file1);
      formData.append("file2", file2);

      const headers: Record<string, string> = {};
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      const response = await fetch("/api/py/compare", {
        method: "POST",
        body: formData,
        headers,
      });

      if (response.status === 401) {
        addToast("error", "Please log in to use this feature.");
        window.location.href = "/login";
        return;
      }

      if (response.status === 429) {
        addToast("warning", "Rate limit exceeded. Please wait a moment.");
        return;
      }

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || body?.error || "Comparison failed");
      }

      const data = await response.json();
      addToast("success", "Analysis complete!");
      onUploadComplete(data);
    } catch (err: any) {
      const msg = err.message || "An error occurred during comparison.";
      setError(msg);
      addToast("error", msg);
    } finally {
      setLoading(false);
    }
  };

  const FileDropzone = ({ file, setFile, title }: { file: File | null, setFile: (f: File) => void, title: string }) => (
    <div 
      className="glass-panel p-8 flex flex-col items-center justify-center border-dashed border-2 hover:border-aura-accent transition-colors cursor-pointer min-h-[250px]"
      onClick={() => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ALLOWED_EXTENSIONS.map(e => `.${e}`).join(",");
        input.onchange = (e) => {
          const target = e.target as HTMLInputElement;
          if (target.files && target.files[0]) {
            handleFileSelect(target.files[0], setFile);
          }
        };
        input.click();
      }}
    >
      {file ? (
        <div className="flex flex-col items-center space-y-4">
          <div className="rounded-full bg-aura-accent/20 p-4">
            <File className="w-10 h-10 text-aura-accent" />
          </div>
          <p className="font-semibold text-lg max-w-[200px] truncate text-center" title={sanitizeFilename(file.name)}>{sanitizeFilename(file.name)}</p>
          <p className="text-sm text-gray-400">{(file.size / 1024).toFixed(2)} KB</p>
        </div>
      ) : (
        <div className="flex flex-col items-center space-y-4 text-gray-400">
          <UploadCloud className="w-12 h-12 mb-2" />
          <p className="font-medium text-lg">{title}</p>
          <p className="text-sm">Click or drag & drop</p>
          <p className="text-xs text-gray-500 mt-2">Support: PDF, DOCX, TXT, PY, JS, TS, Java, C++</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="w-full max-w-5xl mx-auto flex flex-col items-center space-y-8">
      <div className="flex flex-col md:flex-row w-full gap-8">
        <div className="flex-1">
          <FileDropzone file={file1} setFile={setFile1} title="Upload Source File" />
        </div>
        <div className="flex items-center justify-center h-auto">
          <div className="hidden md:flex flex-col items-center space-y-2 opacity-50">
            <div className="w-2 h-2 rounded-full bg-white"></div>
            <div className="w-2 h-2 rounded-full bg-white"></div>
            <div className="w-2 h-2 rounded-full bg-white"></div>
          </div>
        </div>
        <div className="flex-1">
          <FileDropzone file={file2} setFile={setFile2} title="Upload Target File" />
        </div>
      </div>
      
      {error && (
        <div className="flex items-center text-aura-danger space-x-2 bg-aura-danger/10 px-4 py-2 rounded-md w-full">
          <AlertCircle className="w-5 h-5" />
          <span>{error}</span>
        </div>
      )}

      <button 
        onClick={handleCompare}
        disabled={loading || !file1 || !file2}
        className={`w-full max-w-md py-4 rounded-xl font-bold text-lg transition-all
          ${(loading || !file1 || !file2) 
            ? "bg-gray-800 text-gray-500 cursor-not-allowed" 
            : "bg-aura-accent text-white hover:bg-blue-600 hover:shadow-[0_0_20px_rgba(59,130,246,0.5)]"}
        `}
      >
        {loading ? (
          <span className="flex items-center justify-center space-x-3">
            <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            <span>Analyzing contextual intelligence...</span>
          </span>
        ) : (
          "Run Multi-Algorithmic Analysis"
        )}
      </button>
    </div>
  );
}
