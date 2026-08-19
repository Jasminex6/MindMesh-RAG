import React, { useState } from 'react';
import { Sliders, RefreshCw, Database } from 'lucide-react';

export default function SettingsView({ onClearSessions }) {
  const [temperature, setTemperature] = useState(0.1);
  const [topK, setTopK] = useState(5);
  const [strategy, setStrategy] = useState('hybrid_rerank');

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-100/70">
      <header className="bg-white border-b border-slate-300 px-8 py-5 shrink-0 shadow-xs">
        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">System Settings & Parameters</h2>
        <p className="text-sm text-slate-500 font-semibold mt-0.5">Configure clinical decision support model parameters and storage</p>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6 max-w-4xl mx-auto w-full">
        {/* Model Parameters Card */}
        <div className="bg-white border border-slate-300 rounded-2xl p-6 shadow-xs space-y-6">
          <div className="flex items-center gap-2 text-purple-700 font-bold text-base">
            <Sliders className="w-5 h-5" />
            <span>LLM & Retrieval Configuration</span>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm font-semibold text-slate-800">
                <label>LLM Temperature</label>
                <span className="text-purple-700 font-bold">{temperature}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-purple-700 cursor-pointer"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm font-semibold text-slate-800">
                <label>Top-K Retrieval Limit</label>
                <span className="text-purple-700 font-bold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full accent-purple-700 cursor-pointer"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-800 block">Retrieval Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full bg-white border border-slate-300 rounded-xl p-2.5 text-sm font-semibold text-slate-900 outline-none focus:border-purple-600 shadow-2xs"
              >
                <option value="hybrid_rerank">Hybrid RRF + Cross-Encoder Reranker (Default)</option>
                <option value="hybrid">Hybrid RRF Only</option>
                <option value="dense">Dense Vector Search Only</option>
                <option value="bm25">BM25 Keyword Search Only</option>
              </select>
            </div>
          </div>
        </div>

        {/* Data & Storage Management */}
        <div className="bg-white border border-slate-300 rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center gap-2 text-slate-900 font-bold text-base">
            <Database className="w-5 h-5 text-slate-600" />
            <span>Session & Storage Reset</span>
          </div>
          <p className="text-xs text-slate-500 font-semibold">Clear all persisted clinical consultation history stored in browser localStorage.</p>

          <button
            onClick={onClearSessions}
            className="bg-red-50 hover:bg-red-100 text-red-700 border border-red-300 font-bold px-4 py-2.5 rounded-xl text-xs flex items-center gap-2 transition-colors cursor-pointer shadow-2xs"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Clear Local Sessions</span>
          </button>
        </div>
      </div>
    </div>
  );
}
