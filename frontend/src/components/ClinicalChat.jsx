import React, { useState, useRef, useEffect } from 'react';
import { 
  Baby, 
  TrendingUp, 
  AlertTriangle, 
  Stethoscope, 
  Send, 
  ShieldAlert, 
  FileText, 
  ChevronDown, 
  ChevronUp, 
  Lock,
  Loader2
} from 'lucide-react';

export default function ClinicalChat({ session, onSendMessage, isSubmitting }) {
  const [inputQuery, setInputQuery] = useState('');
  const [expandedEvidenceId, setExpandedEvidenceId] = useState(null);
  const messagesEndRef = useRef(null);

  const messages = session ? session.messages || [] : [];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSubmitting]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!inputQuery.trim() || isSubmitting) return;
    onSendMessage(inputQuery.trim());
    setInputQuery('');
  };

  const quickActionCards = [
    {
      id: 'wheezing',
      icon: Baby,
      iconColor: 'text-purple-700 bg-purple-100/80',
      title: 'Child under 5 with wheezing',
      category: 'Diagnostic trial & referral',
      query: 'What are the symptoms and management of a child under 5 with wheezing?',
    },
    {
      id: 'escalation',
      icon: TrendingUp,
      iconColor: 'text-blue-700 bg-blue-100/80',
      title: 'Asthma treatment escalation',
      category: 'Stepwise MART & second-line',
      query: 'What is the recommended treatment escalation for uncontrolled asthma in children?',
    },
    {
      id: 'red_flags',
      icon: AlertTriangle,
      iconColor: 'text-red-700 bg-red-100/80',
      title: 'Red flags requiring referral',
      category: 'Severe admission criteria',
      query: 'What are the red flags requiring urgent hospital referral in acute severe asthma?',
    },
    {
      id: 'inhaler',
      icon: Stethoscope,
      iconColor: 'text-purple-700 bg-purple-100/80',
      title: 'Inhaler technique & education',
      category: 'Spacers & pediatric devices',
      query: 'What are the guideline recommendations for inhaler technique and spacer education?',
    },
  ];

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-100/70">
      {/* Top Header Bar */}
      <header className="bg-white border-b border-slate-300 px-8 py-5 flex items-center justify-between shrink-0 shadow-xs">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Pediatric Asthma CDS</h2>
            <div className="flex items-center gap-1.5">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-50 text-blue-700 border border-blue-300 cursor-pointer hover:bg-blue-100 transition-colors">
                WHO
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-purple-50 text-purple-700 border border-purple-300 cursor-pointer hover:bg-purple-100 transition-colors">
                NICE NG245
              </span>
            </div>
          </div>
          <p className="text-sm text-slate-500 font-semibold mt-0.5">Evidence-grounded clinical decision support</p>
        </div>

        {/* Right Notice Badge */}
        <div className="hidden md:flex items-center gap-2.5 bg-white border border-slate-300 text-xs text-slate-600 rounded-xl p-3 max-w-xs shadow-xs">
          <ShieldAlert className="w-4 h-4 text-blue-700 shrink-0" />
          <div>
            <span className="font-bold text-slate-900 block">Clinical prototype</span>
            <span className="text-[11px] text-slate-500 font-medium leading-tight">Not a substitute for professional diagnosis or emergency care.</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
        {messages.length === 0 ? (
          /* HERO INITIAL STATE */
          <div className="max-w-4xl mx-auto py-8 text-center space-y-8">
            <div className="space-y-2">
              <h3 className="text-3xl font-bold text-slate-900 tracking-tight">How can I help with this patient?</h3>
              <p className="text-slate-600 font-medium text-base max-w-xl mx-auto">
                Ask about symptoms, diagnosis, treatment, inhaler use, or guideline recommendations.
              </p>
            </div>

            {/* 4 Quick Action Cards Grid (100% Clickable Card) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-left">
              {quickActionCards.map((card) => {
                const Icon = card.icon;
                return (
                  <div
                    key={card.id}
                    onClick={() => onSendMessage(card.query)}
                    className="bg-white border border-slate-300 rounded-2xl p-5 cursor-pointer transition-all hover:border-purple-600 hover:shadow-md hover:-translate-y-0.5 flex flex-col justify-between h-48 group select-none shadow-xs"
                  >
                    <div className="space-y-4">
                      <div className={`w-11 h-11 rounded-xl flex items-center justify-center ${card.iconColor}`}>
                        <Icon className="w-5.5 h-5.5" />
                      </div>
                      <div>
                        <h4 className="font-bold text-slate-900 text-sm group-hover:text-purple-700 transition-colors leading-snug">
                          {card.title}
                        </h4>
                        <p className="text-xs text-slate-500 font-semibold mt-1">
                          {card.category}
                        </p>
                      </div>
                    </div>
                    <div className="text-[11px] font-bold text-purple-700 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
                      <span>Ask question →</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          /* CHAT MESSAGE STREAM */
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <div key={idx} className="space-y-4">
                {msg.role === 'user' ? (
                  <div className="flex justify-end">
                    <div className="bg-purple-700 text-white rounded-2xl rounded-tr-xs px-5 py-3.5 max-w-xl shadow-sm text-sm font-semibold leading-relaxed">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-start">
                    <div className="bg-white border border-slate-300 rounded-2xl rounded-tl-xs p-6 max-w-3xl shadow-sm space-y-4 w-full">
                      {msg.response ? (
                        <>
                          {/* Card 1: Recommendation / Clinical Features */}
                          <div className="bg-blue-50/80 border-l-4 border-blue-600 border-y border-r border-slate-300 rounded-r-xl p-4 text-slate-900 text-sm leading-relaxed shadow-2xs">
                            <div className="font-extrabold text-blue-900 text-xs uppercase tracking-wider mb-1">
                              {msg.content.toLowerCase().includes('symptom') ? '1. Clinical Features' : '1. Recommendation'}
                            </div>
                            <div className="font-semibold">{msg.response.recommendation}</div>
                          </div>

                          {/* Card 2: Supporting Evidence */}
                          {msg.response.evidence && msg.response.evidence.length > 0 && (
                            <div className="bg-amber-50/80 border-l-4 border-amber-500 border-y border-r border-slate-300 rounded-r-xl p-4 text-slate-900 text-sm leading-relaxed space-y-2 shadow-2xs">
                              <div className="font-extrabold text-amber-950 text-xs uppercase tracking-wider">
                                2. Supporting Evidence
                              </div>
                              <ul className="space-y-1 text-slate-900 text-xs list-disc pl-4 font-medium">
                                {msg.response.evidence.slice(0, 3).map((ev, i) => (
                                  <li key={i}>{ev.text?.slice(0, 180)}...</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {/* Card 3: Citations & Provenance */}
                          {msg.response.citations && msg.response.citations.length > 0 && (
                            <div className="space-y-2 bg-emerald-50/60 border-l-4 border-emerald-500 border-y border-r border-slate-300 rounded-r-xl p-4 shadow-2xs">
                              <div className="font-extrabold text-emerald-950 text-xs uppercase tracking-wider">
                                3. Citations & Guideline Provenance
                              </div>
                              <div className="space-y-2">
                                {msg.response.citations.map((cite, i) => (
                                  <div key={i} className="text-xs bg-white border border-emerald-300 rounded-lg p-2.5 shadow-2xs">
                                    <div className="flex items-center gap-2 mb-1">
                                      <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold text-white ${cite.verified ? 'bg-emerald-600' : 'bg-red-600'}`}>
                                        {cite.verified ? 'VERIFIED' : 'UNVERIFIED'}
                                      </span>
                                      <span className="font-bold text-slate-900">{cite.claim}</span>
                                    </div>
                                    <div className="text-[11px] text-slate-600 font-medium flex flex-wrap gap-2">
                                      <span>📄 {cite.document}</span>
                                      <span>| Section: {cite.section}</span>
                                      <span>| Page: {cite.page}</span>
                                      <span>| Reranker Score: {cite.score?.toFixed(4)}</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Card 4: Confidence & Safety */}
                          <div className="bg-rose-50/80 border-l-4 border-rose-500 border-y border-r border-slate-300 rounded-r-xl p-4 text-slate-900 text-xs leading-relaxed shadow-2xs">
                            <div className="font-extrabold text-rose-950 text-xs uppercase tracking-wider mb-1">
                              4. Confidence & Safety
                            </div>
                            <div>Confidence: <strong className="text-rose-950">{msg.response.confidence}</strong></div>
                            <div className="text-slate-700 font-medium mt-0.5">{msg.response.safety_message}</div>
                          </div>

                          {/* Evidence Drawer */}
                          {msg.response.evidence && (
                            <div className="pt-2 border-t border-slate-200">
                              <button
                                onClick={() => setExpandedEvidenceId(expandedEvidenceId === idx ? null : idx)}
                                className="flex items-center gap-1.5 text-xs font-bold text-purple-700 hover:text-purple-800 transition-colors cursor-pointer"
                              >
                                <FileText className="w-3.5 h-3.5" />
                                <span>Inspect Retrieved Guideline Passages</span>
                                {expandedEvidenceId === idx ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                              </button>

                              {expandedEvidenceId === idx && (
                                <div className="mt-3 space-y-2 bg-slate-50 p-3 rounded-xl border border-slate-300 text-xs">
                                  {msg.response.evidence.map((ev, i) => (
                                    <div key={i} className="bg-white p-3 rounded-lg border border-slate-300 space-y-1 shadow-2xs">
                                      <div className="flex justify-between text-slate-500 font-semibold">
                                        <span>Chunk: {ev.chunk_id}</span>
                                        <span>Score: {ev.retrieval_score?.toFixed(4)}</span>
                                      </div>
                                      <div className="text-slate-800 font-medium">{ev.text}</div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="text-sm text-slate-900 font-semibold">{msg.content}</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {isSubmitting && (
              <div className="flex justify-start">
                <div className="bg-white border border-slate-300 rounded-2xl p-4 flex items-center gap-3 shadow-xs text-sm font-semibold text-slate-700">
                  <Loader2 className="w-4 h-4 text-purple-700 animate-spin" />
                  <span>Searching WHO & NICE guidelines...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Bottom Prompt Input Bar */}
      <footer className="bg-white/90 backdrop-blur-md border-t border-slate-300 px-8 py-4 shrink-0 shadow-xs">
        <div className="max-w-4xl mx-auto space-y-2">
          <form 
            onSubmit={handleSubmit}
            className="bg-white border border-slate-300 focus-within:border-purple-600 focus-within:ring-2 focus-within:ring-purple-100 shadow-sm rounded-2xl px-4 py-2.5 flex items-center gap-3 transition-all"
          >
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask a clinical question..."
              className="flex-1 bg-transparent border-none outline-none text-slate-900 placeholder:text-slate-400 text-sm font-semibold"
              disabled={isSubmitting}
            />
            <button
              type="submit"
              disabled={!inputQuery.trim() || isSubmitting}
              className="bg-purple-700 hover:bg-purple-800 text-white p-2.5 rounded-xl disabled:opacity-40 transition-colors cursor-pointer shrink-0 shadow-2xs"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>

          <div className="flex items-center justify-center gap-1.5 text-xs text-slate-500 font-medium">
            <Lock className="w-3 h-3 text-slate-400" />
            <span>Answers are grounded in WHO and NICE guidelines with evidence citations.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
