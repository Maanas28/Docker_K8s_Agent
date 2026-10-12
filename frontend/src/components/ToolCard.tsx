import { useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  CircleX,
  Clock,
  Eye,
  LoaderCircle,
  PencilLine,
  Repeat,
  ShieldCheck,
  Trash2,
  TriangleAlert,
  X,
} from "lucide-react";
import type { Decision, ToolItem } from "../types";
import { FileDiff } from "./FileDiff";
import { ToolOutput } from "./ToolOutput";

const RISK = {
  read: { label: "Read-only", icon: Eye },
  change: { label: "Change", icon: PencilLine },
  destructive: { label: "Destructive", icon: Trash2 },
} as const;

const STATUS: Record<ToolItem["status"], { label: string; icon: typeof Check }> = {
  pending: { label: "Needs approval", icon: Clock },
  running: { label: "Running", icon: LoaderCircle },
  done: { label: "Done", icon: Check },
  failed: { label: "Failed", icon: CircleX },
  rejected: { label: "Not run", icon: X },
  expired: { label: "Expired · not run", icon: Clock },
};

function useRemaining(until?: number, active = true) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!until || !active) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [until, active]);
  return until ? Math.max(0, until - now) : null;
}

function useElapsed(since?: number, active = true) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!since || !active) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [since, active]);
  return since ? Math.max(0, Math.round((now - since) / 1000)) : null;
}

// The last lines of a running tool's output, kept scrolled to the bottom like a terminal.
function LiveOutput({ lines, elapsed }: { lines: string[]; elapsed: number | null }) {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines]);
  return (
    <div className="output terminal live">
      <div className="output-head">
        <span className="term-dots"><i /><i /><i /></span>
        <span>Live output · {lines.length} lines{elapsed !== null ? ` · ${elapsed} s` : ""}</span>
      </div>
      <pre ref={ref}>{lines.slice(-200).join("\n")}</pre>
    </div>
  );
}

type Props = { item: ToolItem; onDecide: (id: string, d: Decision, confirm?: string) => void };

export function ToolCard({ item, onDecide }: Props) {
  const [typed, setTyped] = useState("");
  const [total] = useState(() => (item.expiresAt ? item.expiresAt - Date.now() : 0));
  const pending = item.status === "pending";
  // Each finished step is collapsed on its own; click its header to open just that one.
  // A card waiting for a decision is always open, so nothing is approved unseen.
  const [expanded, setExpanded] = useState(false);
  // A tool that streams output (a build) stays open while it runs, so you can watch it.
  const live = item.status === "running" && (item.progress?.length ?? 0) > 0;
  const open = pending || expanded || live;
  const elapsed = useElapsed(item.startedAt, item.status === "running");
  const remaining = useRemaining(item.expiresAt, pending);
  const needsTyping = item.confirmText !== null;
  const typedOk = !needsTyping || typed.trim() === item.confirmText;
  const risk = RISK[item.risk];
  const status = STATUS[item.status];
  const RiskIcon = risk.icon;
  const StatusIcon = status.icon;
  const hasArgs = Object.keys(item.args).length > 0;

  const secs = remaining !== null ? Math.round(remaining / 1000) : 0;
  const countdown = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;

  if (!open) {
    return (
      <button
        className={`step-row risk-${item.risk} is-${item.status}`}
        onClick={() => setExpanded(true)}
        aria-expanded={false}
        title={`${item.tool} · ${risk.label}${item.needsApproval ? "" : item.allowedBy === "session" ? " · allowed this session" : " · always allowed"}`}
      >
        <ChevronRight size={13} className="card-chev" />
        <RiskIcon size={13} className="step-risk" />
        <span className="step-text">{item.summary}</span>
        {item.allowedBy === "session" && <span className="session-badge" title="Allowed for this session"><Repeat size={11} /> session</span>}
        <span className={`step-status is-${item.status}`}>
          <StatusIcon size={12} className={item.status === "running" ? "spin" : undefined} />
          {status.label}
        </span>
      </button>
    );
  }

  return (
    <article className={`tool-card risk-${item.risk} is-${item.status}`}>
      <header
        className={`tool-top${pending ? "" : " toggle"}`}
        onClick={pending ? undefined : () => setExpanded(!expanded)}
        role={pending ? undefined : "button"}
        tabIndex={pending ? undefined : 0}
        aria-expanded={pending ? undefined : expanded}
        onKeyDown={
          pending ? undefined : (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), setExpanded(!expanded))
        }
      >
        {!pending && <ChevronRight size={15} className="card-chev" />}
        <span className="risk-icon"><RiskIcon size={16} /></span>
        <div className="tool-title">
          <span className="tool-summary">{item.summary}</span>
          <span className="tool-sub">
            <code>{item.tool}</code>
            <span className="sep">·</span>
            <span className="risk-label">{risk.label}</span>
            {!item.needsApproval && (
              <>
                <span className="sep">·</span>
                {item.allowedBy === "session" ? (
                  <span className="auto"><Repeat size={12} /> allowed this session</span>
                ) : (
                  <span className="auto"><ShieldCheck size={12} /> always allowed</span>
                )}
              </>
            )}
          </span>
        </div>
        <span className={`status-pill is-${item.status}`}>
          <StatusIcon size={13} className={item.status === "running" ? "spin" : undefined} />
          {status.label}
        </span>
      </header>

      {open && (
        <>
          {item.preview && (pending || !item.result) && (
            <div className="preview">
              {item.preview.reason && (
                <p className="reason"><strong>Why:</strong> {item.preview.reason}</p>
              )}
              {Object.keys(item.preview.facts).length > 0 && (
                <dl className="facts">
                  {Object.entries(item.preview.facts).map(([k, v]) => (
                    <div key={k}>
                      <dt>{k}</dt>
                      <dd>{v}</dd>
                    </div>
                  ))}
                </dl>
              )}
              {item.preview.warnings.map((w) => (
                <p key={w} className="warning"><TriangleAlert size={13} /> {w}</p>
              ))}
              {item.preview.commands && item.preview.commands.length > 0 && (
                <div className="commands">
                  <span className="commands-label">Commands it will run</span>
                  <pre>{item.preview.commands.join("\n")}</pre>
                </div>
              )}
              {item.preview.file && <FileDiff file={item.preview.file} />}
            </div>
          )}
          {item.preview?.file && item.result && (
            <details className="args">
              <summary><ChevronRight size={13} className="chev" /> What was written</summary>
              <FileDiff file={item.preview.file} />
            </details>
          )}

          {hasArgs && (
            <details className="args">
              <summary><ChevronRight size={13} className="chev" /> Arguments</summary>
              <pre>{JSON.stringify(item.args, null, 2)}</pre>
            </details>
          )}

          {pending && (
            <div className="approval">
              {needsTyping && (
                <label className="confirm">
                  <span>Type <code>{item.confirmText}</code> to confirm</span>
                  <input
                    value={typed}
                    onChange={(e) => setTyped(e.target.value)}
                    placeholder={item.confirmText ?? ""}
                    autoComplete="off"
                    spellCheck={false}
                  />
                </label>
              )}
              <div className="approval-actions">
                <button className="btn btn-primary" disabled={!typedOk} onClick={() => onDecide(item.id, "approve", typed)}>
                  <Check size={15} /> Approve
                </button>
                {item.canSessionAllow && (
                  <button
                    className="btn btn-secondary"
                    onClick={() => onDecide(item.id, "session_allow")}
                    title={`Approve this, and don't ask again in this chat to ${item.sessionAllowLabel}. Revoke any time above the message box.`}
                  >
                    <Repeat size={15} /> Allow for this session
                  </button>
                )}
                {item.canAlwaysAllow && (
                  <button className="btn btn-secondary" onClick={() => onDecide(item.id, "always_allow")} title="Don't ask again for this (read-only). Revoke any time.">
                    <ShieldCheck size={15} /> {item.alwaysAllowLabel}
                  </button>
                )}
                <button className="btn btn-ghost-danger" onClick={() => onDecide(item.id, "reject")}>
                  <X size={15} /> Reject
                </button>
                {remaining !== null && <span className="expires">Expires in {countdown}</span>}
              </div>
              {remaining !== null && total > 0 && (
                <div className="expiry-bar"><span style={{ width: `${(remaining / total) * 100}%` }} /></div>
              )}
            </div>
          )}

          {item.decision === "session_allow" && !pending && (
            <p className="tool-note"><Repeat size={13} /> Won't ask again in this chat to {item.sessionAllowLabel}. Revoke it above the message box.</p>
          )}
          {item.decision === "always_allow" && !pending && (
            <p className="tool-note"><ShieldCheck size={13} /> Won't ask again for this. Revoke it under {item.alwaysAllowLabel.startsWith("Always allow reading") ? "Projects" : "Permissions"}.</p>
          )}
          {item.error && <p className="tool-error">{item.error}</p>}
          {live && <LiveOutput lines={item.progress!} elapsed={elapsed} />}
          {item.result && <ToolOutput result={item.result} />}
        </>
      )}
    </article>
  );
}
