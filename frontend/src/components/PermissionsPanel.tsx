import { useCallback, useEffect, useState } from "react";
import { Eye, PencilLine, ShieldCheck, Trash2, X } from "lucide-react";
import type { Risk } from "../types";

type ToolInfo = { name: string; description: string; risk: Risk; always_allowed: boolean; per_project: boolean };

const RISK_ICON = { read: Eye, change: PencilLine, destructive: Trash2 } as const;

export function PermissionsPanel({ onClose }: { onClose: () => void }) {
  const [tools, setTools] = useState<ToolInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/tools");
      setTools(await res.json());
      setError(null);
    } catch {
      setError("Couldn't load tools. Is the backend running?");
    }
  }, []);

  useEffect(() => {
    load();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [load, onClose]);

  const revoke = async (name: string) => {
    await fetch(`/api/permissions/${encodeURIComponent(name)}`, { method: "DELETE" });
    load();
  };

  return (
    <div className="overlay" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label="Tools and permissions">
        <header className="drawer-head">
          <div>
            <h2>Tools &amp; permissions</h2>
            <p>Every tool asks the first time. Read-only tools can be set to “Always allow”. Change and destructive tools always ask.</p>
          </div>
          <button className="icon-btn square" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </header>
        {error && <p className="tool-error">{error}</p>}
        <ul className="tool-list">
          {tools?.map((t) => {
            const Icon = RISK_ICON[t.risk];
            return (
              <li key={t.name} className={`risk-${t.risk}`}>
                <span className="risk-icon"><Icon size={15} /></span>
                <div className="tool-list-body">
                  <code>{t.name}</code>
                  <p>{t.description}</p>
                </div>
                {t.per_project ? (
                  <span className="asks">Per project · see Projects</span>
                ) : t.always_allowed ? (
                  <button className="btn btn-small" onClick={() => revoke(t.name)} title="Ask again next time">
                    <ShieldCheck size={13} /> Revoke
                  </button>
                ) : (
                  <span className="asks">Asks first</span>
                )}
              </li>
            );
          })}
        </ul>
      </aside>
    </div>
  );
}
