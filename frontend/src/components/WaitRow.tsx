import { useEffect, useState } from "react";
import { Hourglass } from "lucide-react";

type Props = { item: { model: string; reason: string; seconds: number; until: number } };

/** "Waiting 23 s for Groq's per-minute limit", counting down; afterwards it says how long it waited. */
export function WaitRow({ item }: Props) {
  const [now, setNow] = useState(Date.now());
  const left = Math.max(0, Math.ceil((item.until - now) / 1000));
  useEffect(() => {
    if (left <= 0) return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [left]);
  return (
    <div className="step-row wait-row" role="status">
      <Hourglass size={13} className="step-risk" />
      <span className="step-text">
        {left > 0 ? `Waiting ${left} s for ${item.reason}…` : `Waited ${item.seconds} s for ${item.reason}.`}
      </span>
    </div>
  );
}
