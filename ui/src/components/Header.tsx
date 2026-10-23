import clsx from "clsx";
import { motion } from "framer-motion";
import { Moon, Sun, Zap } from "lucide-react";
import type { Theme } from "@/hooks/useTheme";
import type { View } from "@/lib/route";
import { Logo, Switch } from "./ui";

const TABS: { view: View; label: string }[] = [
  { view: "explore", label: "Explorer" },
  { view: "benchmark", label: "Benchmark" },
  { view: "method", label: "Method" },
];

export function Header({
  view,
  onView,
  theme,
  onTheme,
  live,
  onLive,
}: {
  view: View;
  onView: (v: View) => void;
  theme: Theme;
  onTheme: () => void;
  live: boolean;
  onLive: (v: boolean) => void;
}) {
  return (
    <header className="sticky top-0 z-40 border-b hairline bg-bg/95">
      <div className="mx-auto flex h-14 max-w-[1760px] items-center gap-2 px-3 sm:gap-4 sm:px-6">
        <a href="#/explore" className="focus-ring flex items-center gap-2.5 pr-2 !no-underline" aria-label="RAGBench home">
          <Logo />
          <div className="hidden items-baseline gap-2 leading-tight sm:flex">
            <span className="text-[18px] font-semibold text-ink">RAGBench</span>
            <span className="text-[14px] italic text-faint">a generation strategy lab</span>
          </div>
        </a>

        <nav aria-label="Sections" className="mx-auto flex h-full sm:gap-3">
          {TABS.map((t) => {
            const active = view === t.view;
            return (
              <button
                key={t.view}
                type="button"
                onClick={() => onView(t.view)}
                aria-current={active ? "page" : undefined}
                className={clsx(
                  "focus-ring relative px-1 text-[14px] transition-colors sm:px-3 sm:text-[15px]",
                  active ? "text-ink" : "text-dim hover:text-ink",
                )}
              >
                <span className="relative">{t.label}</span>
                {active && (
                  <motion.span
                    layoutId="tab-active"
                    className="absolute inset-x-1 bottom-0 h-[2px] bg-ink"
                    transition={{ duration: 0.18 }}
                  />
                )}
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-1.5 sm:gap-3">
          <span
            className={clsx(
              "num hidden items-center gap-1.5 border px-2 py-0.5 text-[12px] md:inline-flex",
              live ? "border-accent text-accent" : "hairline text-dim",
            )}
            aria-live="polite"
          >
            <span className={clsx("h-1.5 w-1.5", live ? "animate-pulse bg-accent" : "bg-faint")} />
            {live ? "live" : "replay"}
          </span>
          <label className="flex cursor-pointer items-center gap-2 text-[14px] text-dim">
            <Zap size={14} className={live ? "text-accent" : ""} />
            <span className="hidden sm:inline">Try it live</span>
            <Switch checked={live} onChange={onLive} label="Try it live with your own Groq key" />
          </label>
          <button
            type="button"
            onClick={onTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="btn-ghost !px-2"
          >
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </div>
      </div>
    </header>
  );
}
