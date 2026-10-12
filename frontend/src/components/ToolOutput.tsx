import { useState } from "react";
import { Check, ChevronRight, CircleCheck, CircleX, Copy, ExternalLink, EyeOff, FileCode, FolderTree } from "lucide-react";
import type { ToolResult } from "../types";

function cell(v: unknown) {
  if (v === null || v === undefined || v === "") return <span className="muted">—</span>;
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

const STATE_CLASS: Record<string, string> = { running: "ok", exited: "bad", paused: "warn", restarting: "warn", dead: "bad" };

function Facts({ facts, padded }: { facts: Record<string, string>; padded?: boolean }) {
  return (
    <dl className={`facts${padded ? " padded" : ""}`}>
      {Object.entries(facts).map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="copy-btn"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        });
      }}
      title="Copy"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export function ToolOutput({ result }: { result: ToolResult }) {
  switch (result.kind) {
    case "table":
      if (result.rows.length === 0) return <p className="output-empty">No results.</p>;
      return (
        <div className="output">
          <div className="output-head">
            <span>{result.rows.length} {result.rows.length === 1 ? "row" : "rows"}</span>
          </div>
          {result.note && <p className="output-note">{result.note}</p>}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>{result.columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {result.rows.map((row, i) => (
                  <tr key={i}>
                    {result.columns.map((c) => (
                      <td key={c}>
                        {c === "state" && typeof row[c] === "string" ? (
                          <span className={`state-badge ${STATE_CLASS[row[c] as string] ?? ""}`}>{row[c] as string}</span>
                        ) : (
                          cell(row[c])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    case "logs": {
      const lines = result.text.split("\n").length;
      return (
        <div className="output terminal">
          <div className="output-head">
            <span className="term-dots"><i /><i /><i /></span>
            <span>{result.title} · {lines} lines</span>
            <CopyButton text={result.text} />
          </div>
          <pre>{result.text}</pre>
        </div>
      );
    }
    case "json": {
      const text = JSON.stringify(result.data, null, 2);
      return (
        <div className="output">
          <div className="output-head">
            <span>{result.title}</span>
            <CopyButton text={text} />
          </div>
          <pre className="json">{text}</pre>
        </div>
      );
    }
    case "facts":
      return (
        <div className="output">
          <div className="output-head facts-head">
            <CircleCheck size={14} />
            <span>{result.title}</span>
          </div>
          {Object.keys(result.facts).length > 0 && <Facts facts={result.facts} padded />}
        </div>
      );
    case "overview":
      return (
        <div className="output">
          <div className="output-head"><FolderTree size={14} /><span>{result.title}</span></div>
          <Facts facts={result.facts} padded />
          <pre className="tree">{result.tree}</pre>
        </div>
      );
    case "context":
      return (
        <div className="output">
          <div className="output-head"><FolderTree size={14} /><span>{result.title}</span></div>
          <Facts facts={result.facts} padded />
          <details className="context-section">
            <summary>Folder tree</summary>
            <pre className="tree">{result.tree}</pre>
          </details>
          {result.files.map((f) => (
            <details key={f.path} className="context-section">
              <summary>
                <FileCode size={13} /> {f.path}
                <span className="muted"> · {f.lines} lines{f.truncated ? " (shortened)" : ""}</span>
                {f.redacted > 0 && <span className="redacted-badge"><EyeOff size={12} /> {f.redacted} hidden</span>}
              </summary>
              <pre className="tree">{f.text}</pre>
            </details>
          ))}
          {result.skipped.length > 0 && (
            <p className="output-note">
              Not included: {result.skipped.map((s) => `${s.path} (${s.reason})`).join("; ")}
            </p>
          )}
        </div>
      );
    case "tree":
      return (
        <div className="output">
          <div className="output-head">
            <FolderTree size={14} />
            <span>{result.title}</span>
            <CopyButton text={result.text} />
          </div>
          <pre className="tree">{result.text}</pre>
        </div>
      );
    case "file": {
      const lines = result.text.split("\n");
      const width = String(result.end_line).length;
      return (
        <div className="output code-file">
          <div className="output-head">
            <FileCode size={14} />
            <span>
              {result.title} · lines {result.start_line}–{result.end_line} of {result.total_lines}
            </span>
            {result.redacted > 0 && (
              <span className="redacted-badge" title="Secret-looking values were masked">
                <EyeOff size={12} /> {result.redacted} hidden
              </span>
            )}
            <CopyButton text={result.text} />
          </div>
          <pre>
            {lines.map((line, i) => (
              <div key={i} className="code-line">
                <span className="ln">{String(result.start_line + i).padStart(width, " ")}</span>
                <span>{line || " "}</span>
              </div>
            ))}
          </pre>
          {result.truncated && <p className="output-note">More lines not shown.</p>}
        </div>
      );
    }
    case "build": {
      const lines = result.log ? result.log.split("\n").length : 0;
      return (
        <div className="output">
          <div className={`output-head facts-head${result.ok ? "" : " is-bad"}`}>
            {result.ok ? <CircleCheck size={14} /> : <CircleX size={14} />}
            <span>{result.title}</span>
          </div>
          {result.error && <p className="tool-error padded">{result.error}</p>}
          {Object.keys(result.facts).length > 0 && <Facts facts={result.facts} padded />}
          {lines > 0 && (
            // A failed build's log is open by default: the error is what you need to see.
            <details className="build-log" open={!result.ok}>
              <summary>
                <ChevronRight size={13} className="chev" /> Build log · {lines} lines
              </summary>
              <div className="output terminal">
                <div className="output-head">
                  <span className="term-dots"><i /><i /><i /></span>
                  <span>docker build</span>
                  <CopyButton text={result.log} />
                </div>
                <pre>{result.log}</pre>
              </div>
            </details>
          )}
        </div>
      );
    }
    case "container": {
      const lines = result.log ? result.log.split("\n").length : 0;
      return (
        <div className="output">
          <div className={`output-head facts-head${result.ok ? "" : " is-bad"}`}>
            {result.ok ? <CircleCheck size={14} /> : <CircleX size={14} />}
            <span>{result.title}</span>
            {result.ok && (
              <a className="open-link" href={result.url} target="_blank" rel="noreferrer">
                Open <ExternalLink size={12} />
              </a>
            )}
          </div>
          {result.error && <p className="tool-error padded">{result.error}</p>}
          {Object.keys(result.facts).length > 0 && <Facts facts={result.facts} padded />}
          {lines > 0 && (
            <details className="build-log" open={!result.ok}>
              <summary>
                <ChevronRight size={13} className="chev" /> Container log · {lines} lines
              </summary>
              <div className="output terminal">
                <div className="output-head">
                  <span className="term-dots"><i /><i /><i /></span>
                  <span>docker logs</span>
                  <CopyButton text={result.log} />
                </div>
                <pre>{result.log}</pre>
              </div>
            </details>
          )}
        </div>
      );
    }
    case "error":
      return <p className="tool-error">{result.message}</p>;
  }
}
