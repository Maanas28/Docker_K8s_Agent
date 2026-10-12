import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Cloud, Cpu } from "lucide-react";
import type { ModelStatus, ModelUsage } from "../types";

const fmt = (n: number) => (n >= 10_000 ? `${Math.round(n / 1000)}k` : n.toLocaleString());

/** The short badge on a model row and on the button: "250 left", "Unlimited", "Used up"… */
export function badge(m: ModelStatus): { text: string; tone: "ok" | "low" | "off" } {
  if (m.local) return { text: "Unlimited", tone: "ok" };
  const u = m.usage;
  if (!u) return { text: "", tone: "ok" };
  if (u.blocked) return { text: "Used up", tone: "off" };
  if (!m.usable) return { text: `${u.calls_left ?? 0} left · reserve`, tone: "off" };
  if (u.calls_left === null) return { text: "No daily limit", tone: "ok" };
  return { text: `${u.calls_left.toLocaleString()} left`, tone: u.calls_left <= u.reserve + 5 ? "low" : "ok" };
}

/** Everything behind the badge, for the tooltip and the group's second line. */
function details(u: ModelUsage): string {
  const parts: string[] = [];
  if (u.requests_limit) parts.push(`${u.requests_used.toLocaleString()} of ${u.requests_limit.toLocaleString()} requests used today`);
  if (u.tokens_limit) parts.push(`${fmt(u.tokens_used)} of ${fmt(u.tokens_limit)} tokens`);
  if (!u.requests_limit && !u.tokens_limit) parts.push(`${u.requests_used} calls today`);
  if (u.requests_per_minute) parts.push(`max ${u.requests_per_minute}/min`);
  if (u.requests_limit || u.tokens_limit) {
    parts.push(u.exact ? "confirmed by the provider" : "counted by this app");
    parts.push(`resets ${u.resets_label}`);
  }
  return parts.join(" · ");
}

type Group = { provider: string; local: boolean; models: ModelStatus[]; shared: boolean };

function groupModels(models: ModelStatus[]): { groups: Group[]; notSetUp: Map<string, string> } {
  const groups: Group[] = [];
  const notSetUp = new Map<string, string>(); // provider -> what to do
  for (const m of models) {
    if (!m.configured) {
      notSetUp.set(m.provider, m.reason);
      continue;
    }
    let g = groups.find((x) => x.provider === m.provider);
    if (!g) groups.push((g = { provider: m.provider, local: m.local, models: [], shared: false }));
    g.models.push(m);
  }
  for (const g of groups) {
    g.shared = g.models.length > 1 && g.models.every((m) => m.pool && m.pool === g.models[0].pool);
  }
  return { groups, notSetUp };
}

type Props = {
  models: ModelStatus[];
  onSelect: (id: string) => void;
  disabled: boolean;
  // Health of the local model (Ollama running and the model pulled); undefined = not known yet.
  localReady: boolean | undefined;
  localProblem?: string;
};

export function ModelPicker({ models, onSelect, disabled, localReady, localProblem }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const selected = models.find((m) => m.selected);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  if (!selected) return null;
  const sel = badge(selected);
  const { groups, notSetUp } = groupModels(models);
  // The header's status dot for the model in use: ready (green), near its limit (yellow), or not usable (red).
  const state = selected.local
    ? (localReady ? "ok" : "bad")
    : !selected.usable ? "bad" : sel.tone === "low" ? "warn" : "ok";
  const stateTitle = selected.local
    ? (localReady ? `Ready: ${selected.model} on this Mac` : localProblem ?? "Checking Ollama…")
    : selected.usable ? `Ready: ${selected.provider} · ${selected.model}` : selected.reason;

  return (
    <div className="model-picker" ref={ref}>
      <button
        className="model-btn"
        onClick={() => setOpen(!open)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={disabled ? `${stateTitle}. Can't switch models while the agent is working.` : stateTitle}
      >
        <span className={`model-dot ${state}`} aria-label={stateTitle} />
        {selected.local ? <Cpu size={14} /> : <Cloud size={14} />}
        <span className="model-btn-label">{selected.label}</span>
        {!selected.local && sel.text && <span className={`model-badge ${sel.tone}`}>{sel.text}</span>}
        <ChevronDown size={14} />
      </button>

      {open && (
        <div className="model-menu" role="listbox" aria-label="Model">
          {groups.map((g) => {
            const usage = g.shared ? g.models[0].usage : null;
            return (
              <div key={g.provider} className="model-group">
                <div className="model-group-head">
                  <span className="model-group-name">
                    {g.local ? <Cpu size={12} /> : <Cloud size={12} />}
                    {g.local ? "On this Mac" : g.provider}
                  </span>
                  {usage && (() => {
                    const shared = badge(g.models[0]);
                    return <span className={`model-badge ${shared.tone}`}>{shared.text}</span>;
                  })()}
                </div>
                {usage && <div className="model-group-detail">Shared by these models · {details(usage)}</div>}
                {g.models.map((m) => {
                  const b = badge(m);
                  const blocked = !m.usable && !m.selected;
                  return (
                    <button
                      key={m.id}
                      role="option"
                      aria-selected={m.selected}
                      className={`model-option${m.selected ? " selected" : ""}`}
                      disabled={blocked}
                      onClick={() => {
                        if (!m.selected) onSelect(m.id);
                        setOpen(false);
                      }}
                      title={[m.model, blocked ? m.reason : m.usage && !g.shared ? details(m.usage) : ""].filter(Boolean).join("\n")}
                    >
                      <span className="model-option-check">{m.selected && <Check size={14} />}</span>
                      <span className="model-option-label">{m.label}</span>
                      {!g.shared && b.text && <span className={`model-badge ${b.tone}`}>{b.text}</span>}
                    </button>
                  );
                })}
              </div>
            );
          })}
          {notSetUp.size > 0 && (
            <div className="model-setup">
              <span>Not set up yet:</span>
              {[...notSetUp].map(([provider, reason]) => (
                <span key={provider} title={reason}>
                  <strong>{provider}</strong> ({reason.replace(" to backend/.env", "")})
                </span>
              ))}
            </div>
          )}
          <div className="model-foot">Cloud models send your code to the provider. Hover a model for details.</div>
        </div>
      )}
    </div>
  );
}
