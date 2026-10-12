import { useState } from "react";
import { ChevronRight, CircleSlash } from "lucide-react";

/** A step that was stopped before running (pre-check failed, repeated attempt, unknown tool…).
 * Shown as one small row like the other steps; click to read why. */
export function NoticeRow({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const colon = text.indexOf(": ");
  const title = colon > 0 && colon < 90 ? text.slice(0, colon) : text.slice(0, 80) + (text.length > 80 ? "…" : "");
  const detail = colon > 0 && colon < 90 ? text.slice(colon + 2) : text;

  return (
    <div className={`notice-row${open ? " open" : ""}`}>
      <button className="step-row is-stopped" onClick={() => setOpen(!open)} aria-expanded={open}>
        <ChevronRight size={13} className="card-chev" />
        <CircleSlash size={13} className="step-risk" />
        <span className="step-text">{title}</span>
        <span className="step-status is-stopped">Stopped</span>
      </button>
      {open && <p className="notice-detail">{detail}</p>}
    </div>
  );
}
