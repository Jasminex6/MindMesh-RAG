import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ClinicalChat from './components/ClinicalChat';
import EvaluationDashboard from './components/EvaluationDashboard';
import ArchitectureView from './components/ArchitectureView';
import SettingsView from './components/SettingsView';

const STORAGE_KEY = 'asthma_cds_sessions';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Initialize Sessions from LocalStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setSessions(parsed);
          setActiveSessionId(parsed[0].id);
          return;
        }
      }
    } catch (e) {
      console.error('Failed to load stored sessions:', e);
    }

    // Default initial new session if empty
    createNewSession();
  }, []);

  // Save Sessions to LocalStorage on Change
  useEffect(() => {
    if (sessions.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
      } catch (e) {
        console.error('Failed to save sessions to localStorage:', e);
      }
    }
  }, [sessions]);

  // Helper: Create New Consultation Session
  const createNewSession = () => {
    const newSession = {
      id: crypto.randomUUID(),
      title: 'New Clinical Session',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      createdAt: new Date().toISOString(),
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setActiveTab('chat');
  };

  // Helper: Select Existing Session
  const selectSession = (id) => {
    setActiveSessionId(id);
    setActiveTab('chat');
  };

  // Helper: Delete Session
  const deleteSession = (id) => {
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== id);
      if (activeSessionId === id && filtered.length > 0) {
        setActiveSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  // Helper: Clear All Sessions
  const clearAllSessions = () => {
    localStorage.removeItem(STORAGE_KEY);
    setSessions([]);
    createNewSession();
  };

  // Active Session Object
  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  // Handle Sending Clinical Message to Backend API
  const handleSendMessage = async (queryText) => {
    if (!activeSessionId || isSubmitting) return;

    setIsSubmitting(true);

    // 1. Add User Message immediately
    const userMsg = { role: 'user', content: queryText };
    const updatedMessages = [...(activeSession?.messages || []), userMsg];

    // Auto-title session from first query if new
    const newTitle = activeSession?.messages?.length === 0 ? queryText.slice(0, 26) : activeSession?.title;

    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeSessionId
          ? { ...s, title: newTitle, messages: updatedMessages }
          : s
      )
    );

    try {
      // 2. Call FastAPI Backend Endpoint
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          conversation_id: activeSessionId,
          chat_history: activeSession?.messages || [],
        }),
      });

      if (!response.ok) {
        let errDetail = `Server returned status ${response.status}`;
        try {
          const errData = await response.json();
          if (errData && errData.detail) {
            errDetail = errData.detail;
          }
        } catch (_) {}
        throw new Error(errDetail);
      }

      const ragResp = await response.json();

      // 3. Add Assistant Structured Response Message
      const assistantMsg = {
        role: 'assistant',
        content: ragResp.recommendation,
        response: ragResp,
      };

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? { ...s, messages: [...s.messages, assistantMsg] }
            : s
        )
      );
    } catch (err) {
      console.error('API call error:', err);
      // Fallback assistant response if API fails
      const fallbackMsg = {
        role: 'assistant',
        content: `Error connecting to backend API (${err.message}). If dev server is running, please restart 'python dev.py' to reload updated backend code.`,
      };
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? { ...s, messages: [...s.messages, fallbackMsg] }
            : s
        )
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50">
      {/* Sidebar (Fixed 280px) */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewConsultation={createNewSession}
        onSelectSession={selectSession}
        onDeleteSession={deleteSession}
      />

      {/* Main Workspace View Switcher */}
      {activeTab === 'chat' && (
        <ClinicalChat
          session={activeSession}
          onSendMessage={handleSendMessage}
          isSubmitting={isSubmitting}
        />
      )}

      {activeTab === 'eval' && <EvaluationDashboard />}

      {activeTab === 'arch' && <ArchitectureView />}

      {activeTab === 'settings' && <SettingsView onClearSessions={clearAllSessions} />}
    </div>
  );
}
