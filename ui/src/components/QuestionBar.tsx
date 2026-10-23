import clsx from "clsx";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useRef } from "react";
import { qLabel, qOriginal } from "@/lib/qlabel";
import { TYPE_LABEL } from "@/lib/strategies";
import type { QuestionOut } from "@/lib/types";

export const typeTone = (type: string): string =>
  ({
    lookup: "text-faint",
    multi_hop: "text-faint",
    out_of_corpus: "text-warn",
    free_text: "text-accent",
  })[type] ?? "text-dim";

export function QuestionBar({
  questions,
  selected,
  onSelect,
}: {
  questions: QuestionOut[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const listRef = useRef<HTMLDivElement>(null);
  const idx = questions.findIndex((q) => q.id === selected);

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-q="${selected}"]`);
    el?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  }, [selected]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || e.metaKey || e.ctrlKey) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const next = questions[idx + (e.key === "ArrowRight" ? 1 : -1)];
        if (next) onSelect(next.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [idx, questions, onSelect]);

  const step = (d: number) => {
    const next = questions[idx + d];
    if (next) onSelect(next.id);
  };

  return (
    <nav aria-label="Benchmark questions" className="flex items-stretch gap-2">
      <button
        type="button"
        onClick={() => step(-1)}
        disabled={idx <= 0}
        aria-label="Previous question"
        className="btn-ghost !px-2 disabled:opacity-30"
      >
        <ChevronLeft size={16} />
      </button>
      <div
        ref={listRef}
        className="no-scrollbar flex flex-1 overflow-x-auto border hairline"
        role="tablist"
      >
        {questions.map((q, i) => {
          const active = q.id === selected;
          return (
            <button
              key={q.id}
              data-q={q.id}
              title={qOriginal(q.id)}
              role="tab"
              aria-selected={active}
              type="button"
              onClick={() => onSelect(q.id)}
              className={clsx(
                "focus-ring relative min-w-[5.25rem] flex-1 shrink-0 px-3 py-2 text-left transition-colors",
                i > 0 && "border-l hairline",
                active ? "bg-ink text-bg" : "text-dim hover:bg-ink/[0.05] hover:text-ink",
              )}
            >
              <span className="flex flex-col">
                <span className="num text-[14px]">{qLabel(q.id)}</span>
                <span className={clsx("text-[12.5px] italic", active ? "text-bg/70" : typeTone(q.type))}>
                  {TYPE_LABEL[q.type]}
                </span>
              </span>
            </button>
          );
        })}
      </div>
      <button
        type="button"
        onClick={() => step(1)}
        disabled={idx >= questions.length - 1}
        aria-label="Next question"
        className="btn-ghost !px-2 disabled:opacity-30"
      >
        <ChevronRight size={16} />
      </button>
    </nav>
  );
}
