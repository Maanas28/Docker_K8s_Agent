import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowUp,
  Ban,
  BookOpen,
  Boxes,
  CircleX,
  Container,
  FileText,
  FolderGit2,
  Plus,
  Repeat,
  ShieldCheck,
  Sparkles,
  Square,
  TriangleAlert,
  Wrench,
} from "lucide-react";
import { ModelPicker } from "./components/ModelPicker";
import { NoticeRow } from "./components/NoticeRow";
import { WaitRow } from "./components/WaitRow";
import { PermissionsPanel } from "./components/PermissionsPanel";
import { ProjectsPanel } from "./components/ProjectsPanel";
import { ToolCard } from "./components/ToolCard";
import type { Health, Kind } from "./types";
import { useChat } from "./useChat";

const KIND_META: Record<Kind, { label: string; icon: typeof Wrench }> = {
  action: { label: "Action on your system", icon: Wrench },
  question: { label: "General answer · no system access", icon: BookOpen },
  unsupported: { label: "Not supported", icon: Ban },
};

const SUGGESTIONS = [
  { icon: BookOpen, title: "Image vs container", text: "What's the difference between an image and a container?" },
  { icon: Container, title: "Running containers", text: "Show my running containers" },
  { icon: FileText, title: "Debug a crash", text: "Did any container exit with an error? Show its logs" },
  { icon: FolderGit2, title: "Understand a project", text: "What does sample-fastapi-app do, and which port does it use?" },
];

function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    const load = () =>
      fetch("/api/health")
        .then((r) => r.json())
        .then(setHealth)
        .catch(() => setHealth(null));
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, []);
  return health;
}

function StatusDot({ ok, label, title }: { ok: boolean | undefined; label: string; title?: string }) {
  return (
    <span className={`status-chip ${ok ? "ok" : "bad"}`} title={title ?? (ok ? `${label}: OK` : `${label}: unavailable`)}>
      <span className="dot" />
      {label}
    </span>
  );
}

function AutoTextarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [props.value]);
  return <textarea ref={ref} rows={1} {...props} />;
}

export default function App() {
  const { items, connected, busy, stopping, send, stop, decide, newChat, sessionGrants, revokeSessionGrant, models, setModel } = useChat();
  const health = useHealth();
  const [draft, setDraft] = useState("");
  const [showPerms, setShowPerms] = useState(false);
  const [showProjects, setShowProjects] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items, busy]);

  const submit = (text = draft) => {
    const t = text.trim();
    if (!t || busy || !connected) return;
    send(t);
    setDraft("");
  };

  const modelOk = health?.ollama && health.model_available;
  const modelTitle = !health?.ollama
    ? "Ollama isn't reachable. Run: ollama serve"
    : !health.model_available
      ? `Model not pulled. Run: ollama pull ${health.model}`
      : `Model ${health.model} is ready`;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            <Boxes size={18} strokeWidth={2.2} />
          </span>
          <div className="brand-text">
            <span className="brand-name">Ops Agent</span>
            <span className="brand-sub">Docker &amp; Kubernetes</span>
          </div>
        </div>
        <div className="top-actions">
          <ModelPicker
            models={models}
            onSelect={setModel}
            disabled={busy || !connected}
            localReady={modelOk}
            localProblem={modelOk ? undefined : modelTitle}
          />
          <div className="status-group">
            <StatusDot ok={health?.docker} label="Docker" title={health?.docker ? "Docker is reachable" : "Docker isn't running"} />
            <StatusDot ok={connected} label={connected ? "Live" : "Offline"} title={connected ? "Connected to backend" : "Reconnecting to backend…"} />
          </div>
          <button className="icon-btn" onClick={() => setShowProjects(true)} title="Projects the agent can read">
            <FolderGit2 size={16} />
            <span className="hide-sm">Projects</span>
          </button>
          <button className="icon-btn" onClick={() => setShowPerms(true)} title="Tools & permissions">
            <ShieldCheck size={16} />
            <span className="hide-sm">Permissions</span>
          </button>
          <button className="icon-btn" onClick={newChat} disabled={busy} title="Start a new chat">
            <Plus size={16} />
            <span className="hide-sm">New chat</span>
          </button>
        </div>
      </header>

      <main className="chat">
        <div className="chat-inner">
          {items.length === 0 && (
            <section className="empty">
              <div className="hero-icon">
                <Sparkles size={26} />
              </div>
              <h1>What do you want to do?</h1>
              <p>Ask in plain English about Docker or your projects. Anything that touches your system waits for your approval.</p>
              <div className="suggestions">
                {SUGGESTIONS.map(({ icon: Icon, title, text }) => (
                  <button key={text} className="suggestion" onClick={() => submit(text)} disabled={!connected}>
                    <span className="suggestion-icon"><Icon size={16} /></span>
                    <span>
                      <span className="suggestion-title">{title}</span>
                      <span className="suggestion-text">{text}</span>
                    </span>
                  </button>
                ))}
              </div>
            </section>
          )}

          {items.map((it) => {
            switch (it.type) {
              case "user":
                return (
                  <div key={it.id} className="row row-user">
                    <div className="bubble-user">{it.text}</div>
                  </div>
                );
              case "assistant":
                return (
                  <div key={it.id} className="row row-assistant">
                    <div className="avatar"><Sparkles size={14} /></div>
                    <div className="assistant-body markdown">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{it.text}</ReactMarkdown>
                    </div>
                  </div>
                );
              case "classification": {
                const { label, icon: Icon } = KIND_META[it.kind];
                return (
                  <div key={it.id} className={`kind kind-${it.kind}`} title={it.reason}>
                    <Icon size={12} />
                    {label}
                  </div>
                );
              }
              case "notice":
                if (it.tone === "info") return <NoticeRow key={it.id} text={it.text} />;
                return (
                  <div key={it.id} className={`notice ${it.tone}`}>
                    <TriangleAlert size={15} />
                    <span>{it.text}</span>
                  </div>
                );
              case "wait":
                return <WaitRow key={it.id} item={it} />;
              case "failure":
                return (
                  <div key={it.id} className="notice failure" role="status">
                    <CircleX size={15} />
                    <div>
                      <strong>{it.summary} failed.</strong>
                      <p className="failure-message">{it.message}</p>
                      {it.retrying && (
                        <p className="failure-next">I'll try to fix it. Each change is shown to you for approval first.</p>
                      )}
                    </div>
                  </div>
                );
              case "tool":
                return <ToolCard key={it.id} item={it} onDecide={decide} />;
            }
          })}

          {busy && !items.some((i) => i.type === "tool" && i.status === "pending") && (
            <div className="row row-assistant" aria-live="polite">
              <div className="avatar pulse"><Sparkles size={14} /></div>
              <div className="thinking">{stopping ? "Stopping after the current step" : "Thinking"}<span className="dots"><span /><span /><span /></span></div>
            </div>
          )}
          <div ref={bottom} />
        </div>
      </main>

      <footer className="composer-wrap">
        {sessionGrants.length > 0 && (
          <div className="session-bar" aria-label="Allowed for this session">
            <Repeat size={13} />
            <span className="session-bar-title">Allowed this session without asking:</span>
            {sessionGrants.map((g) => (
              <span key={`${g.group}:${g.scope}`} className="session-grant">
                {g.label}
                <button type="button" onClick={() => revokeSessionGrant(g)} title="Ask again from now on">Revoke</button>
              </span>
            ))}
          </div>
        )}
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <AutoTextarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={connected ? "Ask anything, or tell me what to do…" : "Connecting to backend…"}
            disabled={!connected}
          />
          {busy ? (
            <button
              className="send stop"
              type="button"
              onClick={stop}
              disabled={stopping || !connected}
              aria-label={stopping ? "Stopping" : "Stop"}
              title={stopping ? "Finishing the current step…" : "Stop: finish the current step and run nothing else"}
            >
              <Square size={14} fill="currentColor" strokeWidth={0} />
            </button>
          ) : (
            <button className="send" type="submit" disabled={!draft.trim() || !connected} aria-label="Send">
              <ArrowUp size={18} strokeWidth={2.4} />
            </button>
          )}
        </form>
        <p className="composer-hint">
          <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line · every action needs your approval{sessionGrants.length > 0 ? " (except what you allowed for this session)" : ""}
        </p>
      </footer>

      {showPerms && <PermissionsPanel onClose={() => setShowPerms(false)} />}
      {showProjects && <ProjectsPanel onClose={() => setShowProjects(false)} />}
    </div>
  );
}
