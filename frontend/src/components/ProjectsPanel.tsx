import { useCallback, useEffect, useState } from "react";
import { FolderGit2, FolderPlus, ShieldCheck, Trash2, TriangleAlert, X } from "lucide-react";
import type { ProjectInfo } from "../types";

async function errorText(res: Response) {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

export function ProjectsPanel({ onClose }: { onClose: () => void }) {
  const [projects, setProjects] = useState<ProjectInfo[] | null>(null);
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/projects");
      if (!res.ok) throw new Error(await errorText(res));
      setProjects(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load projects. Is the backend running?");
    }
  }, []);

  useEffect(() => {
    load();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [load, onClose]);

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!path.trim()) return;
    setSaving(true);
    setError(null);
    const res = await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path.trim(), name: name.trim() || null }),
    });
    setSaving(false);
    if (!res.ok) {
      setError(await errorText(res));
      return;
    }
    setPath("");
    setName("");
    load();
  };

  const remove = async (project: string) => {
    if (!window.confirm(`Remove "${project}"? The agent will no longer be able to read it. Your files are not touched.`)) return;
    await fetch(`/api/projects/${encodeURIComponent(project)}`, { method: "DELETE" });
    load();
  };

  const revoke = async (project: string) => {
    await fetch(`/api/projects/${encodeURIComponent(project)}/read-grant`, { method: "DELETE" });
    load();
  };

  return (
    <div className="overlay" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label="Projects">
        <header className="drawer-head">
          <div>
            <h2>Projects</h2>
            <p>
              The agent can only read folders you add here. Secret files like <code>.env</code> and keys are
              never read, and it can't leave the folder.
            </p>
          </div>
          <button className="icon-btn square" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </header>

        <form className="project-form" onSubmit={add}>
          <label>
            <span>Folder path</span>
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="~/Projects/my-app"
              spellCheck={false}
              autoComplete="off"
            />
          </label>
          <label>
            <span>Name <em>(optional, defaults to the folder name)</em></span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-app" spellCheck={false} autoComplete="off" />
          </label>
          <button className="btn btn-primary" type="submit" disabled={!path.trim() || saving}>
            <FolderPlus size={15} /> {saving ? "Adding…" : "Add project"}
          </button>
        </form>

        {error && <p className="tool-error">{error}</p>}

        <ul className="tool-list">
          {projects?.length === 0 && <li className="empty-row">No projects yet.</li>}
          {projects?.map((p) => (
            <li key={p.name}>
              <span className="risk-icon project-icon"><FolderGit2 size={15} /></span>
              <div className="tool-list-body">
                <code>{p.name}</code>
                <p className="path">{p.path}</p>
                {!p.exists && (
                  <p className="missing"><TriangleAlert size={12} /> Folder not found</p>
                )}
                <p className="grant">
                  {p.reads_always_allowed ? (
                    <>
                      <ShieldCheck size={12} /> Reading always allowed ·{" "}
                      <button className="link-btn" onClick={() => revoke(p.name)}>Revoke</button>
                    </>
                  ) : (
                    "Asks before reading"
                  )}
                </p>
              </div>
              <button className="icon-btn square danger" onClick={() => remove(p.name)} title="Remove project" aria-label={`Remove ${p.name}`}>
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
