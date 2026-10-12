import { FilePlus2, FilePen } from "lucide-react";
import type { PreviewFile } from "../types";

type Row = { kind: "add" | "del" | "ctx" | "hunk"; text: string; oldNo?: number; newNo?: number };

/** Turn a unified diff into rows with old/new line numbers. */
function parseDiff(diff: string): Row[] {
  const rows: Row[] = [];
  let oldNo = 0;
  let newNo = 0;
  for (const line of diff.split("\n")) {
    if (line.startsWith("---") || line.startsWith("+++") || line === "") continue;
    const hunk = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunk) {
      oldNo = Number(hunk[1]);
      newNo = Number(hunk[2]);
      rows.push({ kind: "hunk", text: line });
    } else if (line.startsWith("+")) {
      rows.push({ kind: "add", text: line.slice(1), newNo: newNo++ });
    } else if (line.startsWith("-")) {
      rows.push({ kind: "del", text: line.slice(1), oldNo: oldNo++ });
    } else if (!line.startsWith("\\")) {
      rows.push({ kind: "ctx", text: line.slice(1), oldNo: oldNo++, newNo: newNo++ });
    }
  }
  return rows;
}

export function FileDiff({ file }: { file: PreviewFile }) {
  const rows: Row[] = file.new
    ? file.content.replace(/\n$/, "").split("\n").map((text, i) => ({ kind: "add", text, newNo: i + 1 }))
    : parseDiff(file.diff);
  const added = rows.filter((r) => r.kind === "add").length;
  const removed = rows.filter((r) => r.kind === "del").length;
  const Icon = file.new ? FilePlus2 : FilePen;

  return (
    <div className="output file-diff">
      <div className="output-head">
        <Icon size={14} />
        <span>
          {file.path} · {file.new ? `new file, ${file.lines} lines` : "changes"}
        </span>
        <span className="diff-stat">
          <span className="plus">+{added}</span>
          {!file.new && <span className="minus">−{removed}</span>}
        </span>
      </div>
      <pre>
        {rows.map((r, i) =>
          r.kind === "hunk" ? (
            <div key={i} className="diff-row hunk">{r.text}</div>
          ) : (
            <div key={i} className={`diff-row ${r.kind}`}>
              {!file.new && <span className="ln">{r.oldNo ?? ""}</span>}
              <span className="ln">{r.newNo ?? ""}</span>
              <span className="sign">{r.kind === "add" ? "+" : r.kind === "del" ? "−" : " "}</span>
              <span className="code">{r.text || " "}</span>
            </div>
          ),
        )}
      </pre>
    </div>
  );
}
