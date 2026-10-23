import clsx from "clsx";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Eye, EyeOff, KeyRound, Loader2, Lock, ServerCrash } from "lucide-react";
import { useState } from "react";
import type { Backend, KeyStatus } from "@/hooks/useLive";

export function KeyPanel({
  backend,
  status,
  onKey,
  onVerify,
}: {
  backend: Backend;
  status: KeyStatus;
  onKey: (key: string) => void;
  onVerify: () => void;
}) {
  // The key lives only in this component's state and in the session's ref (memory). It is never
  // written to storage, cookies or the URL, and unmounting the panel discards it.
  const [value, setValue] = useState("");
  const [show, setShow] = useState(false);
  const ready = backend.state === "up" && backend.liveReady;
  const verifying = status.state === "verifying";

  return (
    <motion.section
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: "auto", opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="overflow-hidden border-b hairline bg-elev"
      aria-label="Bring your own Groq key"
    >
      <div className="mx-auto grid max-w-[1760px] gap-6 px-4 py-5 sm:px-6 lg:grid-cols-[1.1fr_1fr]">
        <div>
          <div className="mb-1 flex items-center gap-2">
            <KeyRound size={15} className="text-accent" />
            <h2 className="text-[19px] font-semibold">Bring your own Groq key</h2>
          </div>
          <p className="mb-3 max-w-xl text-[15px] leading-relaxed text-dim">
            Run any benchmark question, or your own, through all five strategies live. Replay stays the default; this
            uses your Groq quota (a full scored run is about 20k tokens).
          </p>

          <form
            className="flex max-w-xl flex-col gap-2 sm:flex-row"
            onSubmit={(e) => {
              e.preventDefault();
              if (value.trim() && ready) onVerify();
            }}
          >
            <div className="relative flex-1">
              <input
                type={show ? "text" : "password"}
                value={value}
                onChange={(e) => {
                  setValue(e.target.value);
                  onKey(e.target.value);
                }}
                placeholder="gsk_…"
                aria-label="Groq API key"
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
                spellCheck={false}
                name="groq-key-not-saved"
                data-1p-ignore="true"
                data-lpignore="true"
                data-form-type="other"
                className="focus-ring num h-10 w-full border bg-bg pl-3 pr-10 text-[14px] hairline placeholder:text-faint"
              />
              <button
                type="button"
                onClick={() => setShow((s) => !s)}
                aria-label={show ? "Hide key" : "Show key"}
                className="focus-ring absolute right-1.5 top-1.5 p-1.5 text-faint hover:text-ink"
              >
                {show ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
            <button type="submit" disabled={!value.trim() || !ready || verifying} className="btn-primary h-10 shrink-0">
              {verifying ? <Loader2 size={15} className="animate-spin" /> : <Lock size={14} />}
              {verifying ? "Verifying…" : "Verify key"}
            </button>
          </form>

          <div className="mt-3 min-h-[28px] max-w-xl text-[14.5px]" aria-live="polite">
            <BackendLine backend={backend} />
            <StatusLine status={status} />
          </div>
        </div>

        <ul className="space-y-2.5 text-[14.5px] leading-snug text-dim">
          {[
            ["Memory only.", "The key lives in this tab's memory. It is never put in localStorage, cookies or the URL, and switching live mode off discards it."],
            ["Per request, over HTTPS.", "It travels in a request header for each run, through this demo's backend, to Groq."],
            ["Never stored or logged.", "The server keeps no copy, no cache entry and no log line containing it, and holds no key of its own. The source is open to check."],
            ["Use a throwaway key.", "Create one at console.groq.com/keys for this and revoke it afterwards."],
          ].map(([head, body]) => (
            <li key={head} className="flex gap-2.5">
              <Lock size={13} className="mt-1 shrink-0 text-accent" />
              <span>
                <b className="font-semibold text-ink">{head}</b> {body}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </motion.section>
  );
}

function BackendLine({ backend }: { backend: Backend }) {
  if (backend.state === "checking")
    return (
      <p className="flex items-center gap-2 text-dim">
        <Loader2 size={13} className="animate-spin" /> Contacting the server…
      </p>
    );
  if (backend.state === "down")
    return (
      <p className="flex items-start gap-2 text-warn">
        <ServerCrash size={14} className="mt-1 shrink-0" />
        <span>
          Can't reach the backend, so live mode is unavailable. Start it with <code className="num bg-ink/10 px-1 text-[13px]">ragbench serve</code>. Replay
          still works.
        </span>
      </p>
    );
  if (!backend.liveReady && backend.liveError)
    return (
      <p className="flex items-start gap-2 text-warn">
        <ServerCrash size={14} className="mt-0.5 shrink-0" />
        <span>
          The server couldn't load its retrieval index ({backend.liveError}), so live runs are unavailable. Check the server log and run{" "}
          <code className="num bg-ink/10 px-1 text-[13px]">ragbench ingest</code>. Replay still works.
        </span>
      </p>
    );
  if (!backend.liveReady)
    return (
      <p className="flex items-center gap-2 text-dim">
        <Loader2 size={13} className="animate-spin" /> Server is warming up its embedding model (usually under a minute)…
      </p>
    );
  return null;
}

function StatusLine({ status }: { status: KeyStatus }) {
  if (status.state === "valid")
    return (
      <p className="flex items-start gap-2 text-good">
        <CheckCircle2 size={15} className="mt-0.5 shrink-0" />
        <span>
          Key accepted. Generator <span className="num">{status.generator}</span> and judge <span className="num">{status.judge}</span> are available.
        </span>
      </p>
    );
  if (status.state === "problem")
    return (
      <p className={clsx("flex items-start gap-2", "text-bad")}>
        <AlertTriangle size={15} className="mt-0.5 shrink-0" />
        <span>
          <b className="font-semibold">{status.title}.</b> {status.hint}
          {status.retryAfterS ? ` Try again in about ${Math.ceil(status.retryAfterS)}s.` : ""}
        </span>
      </p>
    );
  return null;
}
