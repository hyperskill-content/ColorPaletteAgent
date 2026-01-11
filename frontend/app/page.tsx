"use client";

import React, { useState, useRef, useCallback } from "react";
import { Upload, Link as LinkIcon, Copy, Check, Loader2 } from "lucide-react";

interface ColorResult {
  rgb: [number, number, number];
  hex: string;
  percentage: number;
  coordinates: [number, number];
  description: string;
}

interface ProcessResult {
  colors: ColorResult[];
  metadata: {
    agent: string;
    image_size: [number, number];
    method: string;
  };
}

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [url, setUrl] = useState("");
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processImage = async (formData: FormData) => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/process-upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Failed to process image");
    } finally {
      setLoading(false);
    }
  };

  const processUrl = async () => {
    if (!url) return;
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch("http://localhost:8000/process-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      setResult(data);
      setImage(url);
    } catch (error) {
      console.error("Error processing URL:", error);
      alert("Failed to process URL");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => setImage(event.target?.result as string);
      reader.readAsDataURL(file);

      const formData = new FormData();
      formData.append("file", file);
      processImage(formData);
    }
  };

  const copyToClipboard = (hex: string, index: number) => {
    navigator.clipboard.writeText(hex);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <main className="min-h-screen px-6 py-12 md:px-24 md:py-20 max-w-7xl mx-auto font-light">
      <header className="mb-16 text-center md:text-left">
        <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-4">Palette Agent</h1>
        <p className="text-zinc-500 text-lg">AI-powered color extraction with spatial reasoning.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
        {/* Left Side: Upload & Image */}
        <div className="space-y-8">
          <div className="flex flex-col gap-4">
            <div 
              onClick={() => fileInputRef.current?.click()}
              className={`
                relative border-2 border-dashed rounded-3xl p-12 flex flex-col items-center justify-center cursor-pointer
                transition-all duration-300 group
                ${image ? 'border-zinc-200' : 'border-zinc-300 hover:border-black hover:bg-zinc-50'}
              `}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                className="hidden" 
                accept="image/*"
              />
              
              {image ? (
                <div className="relative w-full aspect-auto rounded-xl overflow-hidden shadow-2xl">
                  <img src={image} alt="Uploaded" className="w-full h-auto" />
                  {result && result.colors.map((color, i) => {
                    const [w, h] = result.metadata.image_size;
                    const [x, y] = color.coordinates;
                    return (
                      <div 
                        key={i}
                        className="absolute w-8 h-8 -ml-4 -mt-4 bg-white border-2 border-black rounded-full flex items-center justify-center text-xs font-bold shadow-lg"
                        style={{ left: `${(x / w) * 100}%`, top: `${(y / h) * 100}%` }}
                      >
                        {i + 1}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <>
                  <Upload className="w-12 h-12 mb-4 text-zinc-400 group-hover:text-black transition-colors" />
                  <p className="text-zinc-500 group-hover:text-black transition-colors">Click or drag an image here</p>
                </>
              )}
            </div>

            <div className="flex gap-2 p-2 bg-zinc-100 rounded-2xl">
              <div className="flex-1 flex items-center px-4">
                <LinkIcon className="w-4 h-4 text-zinc-400 mr-2" />
                <input 
                  type="text" 
                  placeholder="Paste image URL..." 
                  className="bg-transparent border-none outline-none w-full text-sm"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </div>
              <button 
                onClick={processUrl}
                disabled={loading || !url}
                className="bg-black text-white px-6 py-2 rounded-xl text-sm font-medium hover:bg-zinc-800 transition-colors disabled:bg-zinc-300"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Process"}
              </button>
            </div>
          </div>
        </div>

        {/* Right Side: Colors */}
        <div className="space-y-6">
          <h2 className="text-xl font-medium mb-6">Generated Palette</h2>
          {!result && !loading && (
            <div className="h-64 border border-zinc-100 rounded-3xl flex items-center justify-center text-zinc-400 italic">
              Upload an image to extract colors
            </div>
          )}
          
          {loading && (
            <div className="space-y-4">
              {[1, 2, 3, 4, 5].map(i => (
                <div key={i} className="h-24 bg-zinc-50 animate-pulse rounded-2xl" />
              ))}
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {result.colors.map((color, i) => (
                <div 
                  key={i}
                  onClick={() => copyToClipboard(color.hex, i)}
                  className="group flex items-center p-4 rounded-2xl border border-zinc-100 bg-white card-hover cursor-pointer"
                >
                  <div className="w-6 text-zinc-400 font-medium mr-4">{i + 1}</div>
                  <div 
                    className="w-16 h-16 rounded-xl shadow-inner mr-6"
                    style={{ backgroundColor: color.hex }}
                  />
                  <div className="flex-1">
                    <div className="text-sm font-medium uppercase tracking-wider">{color.hex}</div>
                    <div className="text-xs text-zinc-500 mt-1 capitalize">{color.description}</div>
                  </div>
                  <div className="text-zinc-300 group-hover:text-black transition-colors">
                    {copiedIndex === i ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                  </div>
                </div>
              ))}
              <div className="mt-8 p-6 bg-zinc-50 rounded-2xl text-[10px] text-zinc-400 font-mono uppercase tracking-widest">
                Agent: {result.metadata.agent} | Method: {result.metadata.method}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

