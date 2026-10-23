import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useState } from "react";
import { Benchmark } from "@/components/Benchmark";
import { Explorer } from "@/components/Explorer";
import { Header } from "@/components/Header";
import { KeyPanel } from "@/components/KeyPanel";
import { Method } from "@/components/Method";
import { Skeleton } from "@/components/ui";
import { useBackend, useLiveSession } from "@/hooks/useLive";
import { useReplayBase } from "@/hooks/useReplay";
import { useTheme } from "@/hooks/useTheme";
import { registerQuestionOrder } from "@/lib/qlabel";
import { useRoute, type View } from "@/lib/route";

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [route, navigate] = useRoute();
  const baseState = useReplayBase();
  const [liveEnabled, setLiveEnabled] = useState(false);
  const session = useLiveSession();
  const backend = useBackend(liveEnabled);

  const setLive = useCallback(
    (on: boolean) => {
      if (!on) session.wipe(); // switching live mode off forgets the key and any live result
      setLiveEnabled(on);
    },
    [session],
  );

  const base = baseState.status === "ready" ? baseState.base : null;
  if (base) registerQuestionOrder(base.questions.map((q) => q.id)); // before children render
  const firstId = base?.questions[0]?.id ?? "";
  const qid = route.qid && base?.questions.some((q) => q.id === route.qid) ? route.qid : firstId;

  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:bg-elev focus:px-3 focus:py-2">
        Skip to content
      </a>
      <Header
        view={route.view}
        onView={(v: View) => navigate({ view: v, qid: v === "explore" ? qid || null : null })}
        theme={theme}
        onTheme={toggleTheme}
        live={liveEnabled}
        onLive={setLive}
      />
      <AnimatePresence initial={false}>
        {liveEnabled && (
          <KeyPanel backend={backend} status={session.keyStatus} onKey={session.setKey} onVerify={() => void session.verify()} />
        )}
      </AnimatePresence>

      <main id="main" className="mx-auto max-w-[1760px] px-4 py-6 sm:px-6 sm:py-8">
        {baseState.status === "loading" && <LoadingShell />}
        {baseState.status === "error" && (
          <div role="alert" className="surface mx-auto mt-16 max-w-xl p-8 text-center">
            <h1 className="text-[22px] font-semibold">Couldn't load the benchmark data</h1>
            <p className="mt-2 text-[15.5px] leading-relaxed text-dim">
              The UI reads static files from <code className="num bg-ink/10 px-1">/replay</code> (create them with{" "}
              <code className="num bg-ink/10 px-1">ragbench export-replay</code>) or, failing that, the API (start it with{" "}
              <code className="num bg-ink/10 px-1">ragbench serve</code>).
            </p>
            <p className="num mt-3 text-[12.5px] text-faint">{baseState.error.message}</p>
          </div>
        )}
        {base && (
          // Fade the new page in; never wait for the old one to fade out. A `mode="wait"` exit
          // can fail to finish while the live Explorer is animating, which left a blank screen.
          <motion.div
            key={route.view}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
          >
            {route.view === "explore" && (
              <Explorer
                base={base}
                qid={qid}
                onSelect={(id) => navigate({ view: "explore", qid: id })}
                liveEnabled={liveEnabled}
                session={session}
                backend={backend}
              />
            )}
            {route.view === "benchmark" &&
              (base.summary ? (
                <Benchmark summary={base.summary} onPick={(id) => navigate({ view: "explore", qid: id })} />
              ) : (
                <p className="surface p-8 text-center text-dim">
                  No aggregate report found. Run <code className="num">ragbench analyze</code> then <code className="num">ragbench export-replay</code>.
                </p>
              ))}
            {route.view === "method" && <Method />}
          </motion.div>
        )}
      </main>

      <footer className="mx-auto max-w-[1760px] px-4 pb-10 pt-4 text-[14px] italic leading-relaxed text-faint sm:px-6">
        <div className="border-t pt-5 hairline">
          Replay shows real recorded results from the benchmark; nothing is simulated.
        </div>
      </footer>
    </div>
  );
}

function LoadingShell() {
  return (
    <div className="space-y-10" aria-busy="true" aria-label="Loading">
      <Skeleton className="h-52 w-full" />
      <Skeleton className="h-14 w-full" />
      <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(252px,1fr))]">
        {[0, 1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-80" />
        ))}
      </div>
    </div>
  );
}
