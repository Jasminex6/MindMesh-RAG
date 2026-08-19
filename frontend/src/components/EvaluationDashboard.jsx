import React from 'react';
import { Download, TrendingUp, CheckCircle, Clock, Award } from 'lucide-react';

export default function EvaluationDashboard() {
  const kpiCards = [
    { title: 'Faithfulness', value: '92.4%', delta: '↑ 4.2% vs last run', icon: CheckCircle, color: 'text-emerald-700 bg-emerald-100/80' },
    { title: 'Context Recall', value: '89.7%', delta: '↑ 3.1% vs last run', icon: Award, color: 'text-purple-700 bg-purple-100/80' },
    { title: 'Answer Relevance', value: '94.1%', delta: '↑ 5.3% vs last run', icon: TrendingUp, color: 'text-blue-700 bg-blue-100/80' },
    { title: 'Avg Latency', value: '3.2s', delta: '↓ 0.6s vs last run', icon: Clock, color: 'text-amber-700 bg-amber-100/80' },
  ];

  const datasets = [
    { name: 'Asthma-NG245', faithfulness: '90.1%', recall: '90.2%', relevance: '95.0%' },
    { name: 'WHO-Guidelines', faithfulness: '91.4%', recall: '88.7%', relevance: '93.2%' },
    { name: 'PedsQA', faithfulness: '92.8%', recall: '89.9%', relevance: '94.1%' },
    { name: 'Internal Test Set', faithfulness: '92.4%', recall: '89.7%', relevance: '94.1%' },
  ];

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-100/70">
      <header className="bg-white border-b border-slate-300 px-8 py-5 flex items-center justify-between shrink-0 shadow-xs">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Evaluation & Benchmarks Dashboard</h2>
          <p className="text-sm text-slate-500 font-semibold mt-0.5">Track system performance, quality, and benchmark results</p>
        </div>
        <button className="bg-white border border-slate-300 hover:bg-slate-50 text-slate-800 font-bold px-4 py-2 rounded-xl text-sm flex items-center gap-2 shadow-2xs transition-colors cursor-pointer">
          <Download className="w-4 h-4 text-slate-600" />
          <span>Export Report</span>
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6 max-w-6xl mx-auto w-full">
        {/* KPI Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {kpiCards.map((kpi, idx) => {
            const Icon = kpi.icon;
            return (
              <div key={idx} className="bg-white border border-slate-300 rounded-2xl p-5 shadow-xs space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-extrabold text-slate-500 uppercase tracking-wider">{kpi.title}</span>
                  <div className={`w-8.5 h-8.5 rounded-lg flex items-center justify-center ${kpi.color}`}>
                    <Icon className="w-4.5 h-4.5" />
                  </div>
                </div>
                <div className="text-3xl font-extrabold text-slate-900 tracking-tight">{kpi.value}</div>
                <div className="text-xs font-bold text-emerald-600">{kpi.delta}</div>
              </div>
            );
          })}
        </div>

        {/* 2-Column Overview & Table */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Bars */}
          <div className="bg-white border border-slate-300 rounded-2xl p-6 shadow-xs space-y-5">
            <h3 className="font-bold text-slate-900 text-base">Performance Overview</h3>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold text-slate-800">
                  <span>Faithfulness (Answer grounded in evidence)</span>
                  <span>92%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 border border-slate-200">
                  <div className="bg-purple-700 h-2 rounded-full w-[92%]"></div>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold text-slate-800">
                  <span>Context Recall (Retrieved relevant context)</span>
                  <span>90%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 border border-slate-200">
                  <div className="bg-blue-600 h-2 rounded-full w-[90%]"></div>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold text-slate-800">
                  <span>Answer Relevance (Useful to clinician)</span>
                  <span>94%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 border border-slate-200">
                  <div className="bg-purple-700 h-2 rounded-full w-[94%]"></div>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold text-slate-800">
                  <span>Groundedness (Citations coverage)</span>
                  <span>93%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 border border-slate-200">
                  <div className="bg-emerald-600 h-2 rounded-full w-[93%]"></div>
                </div>
              </div>
            </div>
          </div>

          {/* Dataset Results Table */}
          <div className="bg-white border border-slate-300 rounded-2xl p-6 shadow-xs space-y-4">
            <h3 className="font-bold text-slate-900 text-base">Benchmark Results</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-300 text-slate-500 font-extrabold uppercase tracking-wider">
                    <th className="py-2.5 px-3">Dataset</th>
                    <th className="py-2.5 px-3">Faithfulness</th>
                    <th className="py-2.5 px-3">Context Recall</th>
                    <th className="py-2.5 px-3">Relevance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 font-semibold text-slate-800">
                  {datasets.map((ds, idx) => (
                    <tr key={idx} className="hover:bg-slate-50 transition-colors">
                      <td className="py-3 px-3 font-extrabold text-slate-900">{ds.name}</td>
                      <td className="py-3 px-3 text-purple-700 font-extrabold">{ds.faithfulness}</td>
                      <td className="py-3 px-3">{ds.recall}</td>
                      <td className="py-3 px-3">{ds.relevance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
