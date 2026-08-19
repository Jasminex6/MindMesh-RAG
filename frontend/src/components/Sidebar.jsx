import React from 'react';
import { 
  Shield, 
  Plus, 
  MessageSquare, 
  BarChart3, 
  Cpu, 
  Settings, 
  Trash2
} from 'lucide-react';

export default function Sidebar({ 
  activeTab, 
  setActiveTab, 
  sessions, 
  activeSessionId, 
  onNewConsultation, 
  onSelectSession, 
  onDeleteSession 
}) {

  const navItems = [
    { id: 'chat', label: 'Clinical Chat', icon: MessageSquare },
    { id: 'eval', label: 'Evaluation & Benchmarks', icon: BarChart3 },
    { id: 'arch', label: 'System Architecture', icon: Cpu },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-[280px] bg-white border-r border-slate-300 h-screen flex flex-col justify-between p-4 shrink-0 select-none shadow-xs">
      <div className="space-y-6">
        {/* 1. Brand Header */}
        <div className="flex items-center gap-3 px-1">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-900 to-purple-700 flex items-center justify-center text-white shadow-md">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 leading-tight">AsthmaCDS</h1>
            <p className="text-xs text-slate-500 font-semibold">Pediatric Asthma CDS</p>
          </div>
        </div>

        {/* 2. Action Button: + New Consultation */}
        <button
          onClick={onNewConsultation}
          className="w-full bg-purple-700 hover:bg-purple-800 text-white font-semibold py-2.5 px-4 rounded-xl flex items-center justify-center gap-2 shadow-sm transition-all cursor-pointer"
        >
          <Plus className="w-5 h-5 text-white" />
          <span>New Consultation</span>
        </button>

        {/* 3. Navigation Links */}
        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-purple-50 text-purple-700 font-bold border border-purple-300 shadow-2xs'
                    : 'text-slate-700 hover:bg-slate-100 font-semibold'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-purple-700' : 'text-slate-600'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* 4. Recent Consultations Section (Dynamic State) */}
        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider px-1 mb-2">
            Recent Consultations
          </h3>
          <div className="space-y-1 max-h-[220px] overflow-y-auto pr-1">
            {sessions.length === 0 ? (
              <div className="text-xs text-slate-400 px-2 py-3 text-center bg-slate-50 rounded-lg border border-dashed border-slate-300 font-medium">
                No previous sessions
              </div>
            ) : (
              sessions.map((session) => {
                const isActive = session.id === activeSessionId;

                return (
                  <div
                    key={session.id}
                    className={`group flex items-center justify-between px-3 py-2 rounded-xl text-xs transition-all cursor-pointer ${
                      isActive
                        ? 'bg-purple-50 border border-purple-300 text-purple-950 font-bold shadow-2xs'
                        : 'text-slate-800 hover:bg-slate-100 border border-slate-200'
                    }`}
                    onClick={() => onSelectSession(session.id)}
                  >
                    <div className="flex items-center gap-2.5 truncate min-w-0 flex-1 pr-1">
                      <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-purple-700' : 'text-slate-500'}`} />
                      <div className="truncate min-w-0">
                        <div className="truncate font-semibold text-xs leading-tight">{session.title}</div>
                        <div className="text-[10px] text-slate-500 font-medium leading-tight">{session.timestamp}</div>
                      </div>
                    </div>

                    {/* Direct Inline Trash Delete Button (No Popups, No Scrollbars) */}
                    <button
                      title="Delete consultation"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(session.id);
                      }}
                      className="opacity-60 group-hover:opacity-100 p-1.5 rounded-lg text-slate-400 hover:text-red-600 hover:bg-red-50 transition-all shrink-0 cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* 5. Sidebar Footer: System Online Status & Guideline Badges */}
      <div className="pt-4 border-t border-slate-300 space-y-3">
        <div className="flex items-center gap-2 px-2.5 py-2 bg-white border border-slate-300 rounded-xl shadow-2xs">
          <div className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
          </div>
          <div>
            <div className="text-xs font-bold text-slate-900">System Online</div>
            <div className="text-[10px] text-slate-500 font-medium">All services operational</div>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-300">
            WHO
          </span>
          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-purple-50 text-purple-700 border border-purple-300">
            NICE NG245
          </span>
        </div>
      </div>
    </aside>
  );
}
