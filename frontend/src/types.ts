export type Risk = "read" | "change" | "destructive";
export type Kind = "action" | "question" | "unsupported";

export type ToolResult =
  | { kind: "table"; columns: string[]; rows: Record<string, unknown>[]; note?: string }
  | { kind: "logs"; title: string; text: string }
  | { kind: "json"; title: string; data: unknown }
  | { kind: "facts"; title: string; facts: Record<string, string> }
  | { kind: "overview"; title: string; facts: Record<string, string>; tree: string }
  | { kind: "tree"; title: string; text: string }
  | {
      kind: "context";
      title: string;
      facts: Record<string, string>;
      tree: string;
      files: { path: string; text: string; lines: number; truncated: boolean; redacted: number }[];
      skipped: { path: string; reason: string }[];
    }
  | {
      kind: "file";
      title: string;
      text: string;
      start_line: number;
      end_line: number;
      total_lines: number;
      truncated: boolean;
      redacted: number;
    }
  | {
      kind: "build";
      ok: boolean;
      title: string;
      error?: string;
      diagnosis?: string[];
      facts: Record<string, string>;
      log: string;
    }
  | {
      kind: "container";
      ok: boolean;
      title: string;
      url: string;
      error?: string;
      diagnosis?: string[];
      facts: Record<string, string>;
      log: string;
    }
  | { kind: "error"; message: string };

export type PreviewFile = { path: string; new: boolean; content: string; diff: string; lines: number };

export type Preview = {
  facts: Record<string, string>;
  warnings: string[];
  file?: PreviewFile;
  // Commands the action will run on this machine (a build's RUN lines).
  commands?: string[];
  // Why the model wants to do this (e.g. which error a file change fixes).
  reason?: string;
};

export type ToolStatus = "pending" | "running" | "done" | "failed" | "rejected" | "expired";

export type ToolItem = {
  type: "tool";
  id: string;
  tool: string;
  risk: Risk;
  summary: string;
  args: Record<string, unknown>;
  status: ToolStatus;
  needsApproval: boolean;
  canAlwaysAllow: boolean;
  alwaysAllowLabel: string;
  canSessionAllow: boolean;
  sessionAllowLabel: string | null;
  // Why it ran without a card: a stored "always allow" (reads) or a grant for this session.
  allowedBy?: "always" | "session";
  confirmText: string | null;
  expiresAt?: number;
  preview?: Preview;
  decision?: string;
  error?: string;
  result?: ToolResult;
  // Live output while a long tool (a build) runs, newest last.
  progress?: string[];
  startedAt?: number;
};

export type ChatItem =
  | { type: "user"; id: string; text: string }
  | { type: "assistant"; id: string; text: string }
  | { type: "classification"; id: string; kind: Kind; reason: string }
  | { type: "notice"; id: string; text: string; tone: "info" | "error" }
  // A step that ran and failed, announced by the backend (not the model) so it's never hidden.
  | { type: "failure"; id: string; summary: string; message: string; retrying: boolean }
  // The agent is waiting out a provider's per-minute limit.
  | { type: "wait"; id: string; model: string; reason: string; seconds: number; until: number }
  | ToolItem;

export type Decision = "approve" | "always_allow" | "session_allow" | "reject";

// "Allow for this session": a group of tools on one project that runs without asking until the chat ends.
export type SessionGrant = { group: string; scope: string; label: string };

// The model dropdown (backend: app/models.py statuses()).
export type ModelUsage = {
  day: string;
  requests_used: number;
  requests_limit: number | null;
  requests_left: number | null;
  tokens_used: number;
  tokens_limit: number | null;
  tokens_left: number | null;
  tokens_estimated: number;
  average_tokens_per_call: number;
  calls_left: number | null; // null = no daily limit
  blocked: boolean;
  synced_at: string | null;
  resets_at: string;
  resets_label: string;
  requests_per_minute: number | null;
  tokens_per_minute: number | null;
  exact: boolean; // the provider itself confirmed today's count
  reserve: number;
};

export type ModelStatus = {
  id: string;
  label: string;
  provider: string;
  model: string;
  local: boolean;
  configured: boolean; // false = its provider has no API key yet
  pool: string | null; // models with the same pool share one limit
  usable: boolean;
  reason: string;
  selected: boolean;
  usage: ModelUsage | null;
};

export type Health = {
  ollama: boolean;
  model_available: boolean;
  model: string;
  docker: boolean;
  // The allowed Kubernetes context (agent.toml [kubernetes]) and whether it answers.
  kubernetes?: { ok: boolean; context: string | null };
};

export type ProjectInfo = {
  name: string;
  path: string;
  source: string;
  created_at: string;
  exists: boolean;
  reads_always_allowed: boolean;
};
