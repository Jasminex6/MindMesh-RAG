import React from 'react';
import { ArrowRight, Globe, Zap, Dna, Database, Bot, BookOpen } from 'lucide-react';

export default function ArchitectureView() {
  const techStack = [
    { title: 'Interface — Streamlit / React', desc: 'Modern web application framework', icon: Globe, color: 'text-purple-700 bg-purple-100/80' },
    { title: 'Backend API — FastAPI', desc: 'High-performance REST service layer', icon: Zap, color: 'text-blue-700 bg-blue-100/80' },
    { title: 'Embeddings — Nomic Embed Text', desc: 'Dense vector embeddings (768-dim)', icon: Dna, color: 'text-emerald-700 bg-emerald-100/80' },
    { title: 'Vector DB — ChromaDB', desc: 'High-speed vector storage & similarity search', icon: Database, color: 'text-indigo-700 bg-indigo-100/80' },
    { title: 'LLM Inference — Llama 3.2 Ollama', desc: 'Grounded clinical answer generation', icon: Bot, color: 'text-purple-700 bg-purple-100/80' },
    { title: 'Knowledge Base — WHO & NICE NG245', desc: 'Official pediatric asthma guidelines', icon: BookOpen, color: 'text-amber-700 bg-amber-100/80' },
  ];

  const pipelineNodes = [
    { title: 'Clinical Question', sub: 'User Input' },
    { title: 'Query Processing', sub: 'Intent & Preprocessing' },
    { title: 'Retriever', sub: 'Semantic Search' },
    { title: 'Knowledge Base', sub: 'WHO + NICE' },
    { title: 'LLM Generation', sub: 'Llama 3.2 Ollama' },
    { title: 'Grounded Answer', sub: 'With Citations' },
  ];

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-100/70">
      <header className="bg-white border-b border-slate-300 px-8 py-5 shrink-0 shadow-xs">
        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">System Architecture & Tech Stack</h2>
        <p className="text-sm text-slate-500 font-semibold mt-0.5">High-level overview of the RAG pipeline and system components</p>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6 max-w-6xl mx-auto w-full">
        {/* Pipeline Box Flow Card */}
        <div className="bg-white border border-slate-300 rounded-2xl p-6 shadow-xs space-y-4">
          <h3 className="font-bold text-slate-900 text-base">RAG Pipeline Architecture</h3>
          <div className="flex items-center justify-between gap-2 overflow-x-auto py-4 px-2">
            {pipelineNodes.map((node, idx) => (
              <React.Fragment key={idx}>
                <div className="bg-slate-50 border border-slate-300 rounded-xl p-3 text-center min-w-[125px] shrink-0 shadow-2xs">
                  <div className="font-bold text-slate-900 text-xs">{node.title}</div>
                  <div className="text-[10px] text-slate-500 font-semibold mt-0.5">{node.sub}</div>
                </div>
                {idx < pipelineNodes.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-purple-700 shrink-0 font-bold" />
                )}
              </React.Fragment>
            ))}
          </div>
          <div className="text-center text-xs text-slate-600 font-semibold pt-3 border-t border-slate-200">
            <strong>Nomic Embeddings:</strong> Text embeddings model
          </div>
        </div>

        {/* Tech Stack Cards Grid */}
        <div className="space-y-4">
          <h3 className="font-bold text-slate-900 text-base">Tech Stack Components</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {techStack.map((item, idx) => {
              const Icon = item.icon;
              return (
                <div key={idx} className="bg-white border border-slate-300 rounded-2xl p-5 shadow-xs flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${item.color}`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-900 text-sm">{item.title}</h4>
                    <p className="text-xs text-slate-500 font-semibold mt-0.5">{item.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
