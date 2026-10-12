import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatItem, Decision, ModelStatus, SessionGrant, ToolItem } from "./types";

const MAX_PROGRESS = 2000;

let counter = 0;
const uid = () => `i${++counter}`;

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/chat`;
}

export function useChat() {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [sessionGrants, setSessionGrants] = useState<SessionGrant[]>([]);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  // True between pressing Stop and the server's "done": the current step is finishing.
  const [stopping, setStopping] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  const updateTool = useCallback((id: string, patch: Partial<ToolItem>) => {
    setItems((prev) => prev.map((it) => (it.type === "tool" && it.id === id ? { ...it, ...patch } : it)));
  }, []);

  const onEvent = useCallback(
    (ev: any) => {
      switch (ev.type) {
        case "classification":
          setItems((p) => [...p, { type: "classification", id: uid(), kind: ev.kind, reason: ev.reason }]);
          break;
        case "token":
          setItems((p) => {
            const last = p[p.length - 1];
            if (last?.type === "assistant") return [...p.slice(0, -1), { ...last, text: last.text + ev.text }];
            return [...p, { type: "assistant", id: uid(), text: ev.text }];
          });
          break;
        case "assistant_rewrite":
          // The model wrote a tool call as text; the backend is running it, so hide that text.
          setItems((p) => {
            const last = p[p.length - 1];
            if (last?.type !== "assistant" || !last.text.endsWith(ev.from)) return p;
            const text = last.text.slice(0, last.text.length - ev.from.length) + ev.to;
            return text.trim() ? [...p.slice(0, -1), { ...last, text }] : p.slice(0, -1);
          });
          break;
        case "approval_request":
        case "tool_started":
          setItems((p) => [
            ...p,
            {
              type: "tool",
              id: ev.id,
              tool: ev.tool,
              risk: ev.risk,
              summary: ev.summary,
              args: ev.args,
              needsApproval: ev.type === "approval_request",
              status: ev.type === "approval_request" ? "pending" : "running",
              canAlwaysAllow: Boolean(ev.can_always_allow),
              alwaysAllowLabel: ev.always_allow_label ?? "Always allow",
              canSessionAllow: Boolean(ev.can_session_allow),
              sessionAllowLabel: ev.session_allow_label ?? null,
              allowedBy: ev.allowed_by,
              confirmText: ev.confirm_text ?? null,
              expiresAt: ev.expires_in ? Date.now() + ev.expires_in * 1000 : undefined,
              preview: ev.preview,
              startedAt: ev.type === "tool_started" ? Date.now() : undefined,
            },
          ]);
          break;
        case "approval_decision":
          updateTool(ev.id, {
            decision: ev.decision,
            error: undefined,
            status: ev.decision === "reject" ? "rejected" : ev.decision === "expired" ? "expired" : "running",
            startedAt: Date.now(),
          });
          break;
        case "tool_progress":
          // Batched output lines of a running tool. Keep the newest MAX_PROGRESS so a long build
          // can't grow the page without bound (the full log comes with the result).
          setItems((p) =>
            p.map((it) =>
              it.type === "tool" && it.id === ev.id
                ? { ...it, progress: [...(it.progress ?? []), ...ev.lines].slice(-MAX_PROGRESS) }
                : it,
            ),
          );
          break;
        case "approval_error":
          updateTool(ev.id, { error: ev.message });
          break;
        case "tool_result":
          updateTool(ev.id, { status: ev.ok ? "done" : "failed", result: ev.result });
          break;
        case "models":
          setModels(ev.models);
          break;
        case "rate_wait":
          setItems((p) => [
            ...p,
            { type: "wait", id: uid(), model: ev.model, reason: ev.reason, seconds: ev.seconds, until: Date.now() + ev.seconds * 1000 },
          ]);
          break;
        case "session_grants":
          setSessionGrants(ev.grants);
          break;
        case "step_failed":
          setItems((p) => [
            ...p,
            { type: "failure", id: uid(), summary: ev.summary, message: ev.message, retrying: Boolean(ev.retrying) },
          ]);
          break;
        case "tool_refused":
          setItems((p) => [...p, { type: "notice", id: uid(), text: ev.message, tone: "info" }]);
          break;
        case "error":
          setItems((p) => [...p, { type: "notice", id: uid(), text: ev.message, tone: "error" }]);
          break;
        case "done":
          setBusy(false);
          setStopping(false);
          break;
      }
    },
    [updateTool],
  );

  useEffect(() => {
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      const sock = new WebSocket(wsUrl());
      ws.current = sock;
      sock.onopen = () => setConnected(true);
      sock.onmessage = (m) => onEvent(JSON.parse(m.data));
      sock.onclose = () => {
        setConnected(false);
        setBusy(false);
        setSessionGrants([]); // grants live in the server session, which ends with the connection
        setStopping(false);
        // The server rejects anything pending when the connection drops.
        setItems((p) =>
          p.map((it) =>
            it.type === "tool" && it.status === "pending"
              ? { ...it, status: "rejected", error: "Connection lost, so this was not run." }
              : it,
          ),
        );
        if (!closed) retry = setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      ws.current?.close();
    };
  }, [onEvent]);

  const send = useCallback((text: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    setItems((p) => [...p, { type: "user", id: uid(), text }]);
    setBusy(true);
    ws.current.send(JSON.stringify({ type: "user_message", text }));
  }, []);

  const decide = useCallback((id: string, decision: Decision, confirmText?: string) => {
    ws.current?.send(JSON.stringify({ type: "approval", id, decision, confirm_text: confirmText }));
  }, []);

  // Closing the socket ends the server-side session (and its history); it reconnects fresh.
  const newChat = useCallback(() => {
    setItems([]);
    ws.current?.close();
  }, []);

  const setModel = useCallback((id: string) => {
    ws.current?.send(JSON.stringify({ type: "set_model", id }));
  }, []);

  const revokeSessionGrant = useCallback((grant: SessionGrant) => {
    ws.current?.send(JSON.stringify({ type: "revoke_session_grant", group: grant.group, scope: grant.scope }));
  }, []);

  // Cooperative stop: the server finishes the current step and starts nothing new.
  const stop = useCallback(() => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    setStopping(true);
    ws.current.send(JSON.stringify({ type: "stop" }));
  }, []);

  return { items, connected, busy, stopping, send, stop, decide, newChat, sessionGrants, revokeSessionGrant, models, setModel };
}
