import clsx from "clsx";
import { motion } from "framer-motion";
import { FileText } from "lucide-react";
import { useState } from "react";
import { num } from "@/lib/format";
import type { ChunkOut } from "@/lib/types";
import { useHighlight } from "./highlight";
import { Badge, Tip } from "./ui";

/** "Title (Lewis et al.)" -> { title: "Title", by: "Lewis et al." } */
export function splitTitle(raw: string | null | undefined, fallback: string) {
  const t = raw ?? fallback;
  const m = t.match(/^(.*?)\s*\(([^()]*)\)\s*$/);
  return m ? { title: m[1], by: m[2] } : { title: t, by: "" };
}

function Passage({ chunk }: { chunk: ChunkOut }) {
  const hl = useHighlight();
  const [open, setOpen] = useState(false);
  const active = hl.passage === chunk.rank;
  const { title, by } = splitTitle(chunk.doc_title, chunk.doc_id);
  return (
    <motion.li
      layout
      className={clsx(
        "relative flex min-w-0 flex-col gap-2 border p-3 transition-colors duration-150",
        active ? "border-accent bg-accent/10" : "hairline",
      )}
      aria-label={`Passage ${chunk.rank}`}
    >
      <div className="flex items-start gap-2.5">
        <span
          className={clsx(
            "num grid h-6 w-6 shrink-0 place-items-center text-[13px] transition-colors",
            active ? "bg-accent text-bg" : "border border-rule text-dim",
          )}
        >
          {chunk.rank}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <FileText size={12} className="shrink-0 text-faint" />
            <span className="truncate text-[14px] font-semibold text-ink" title={title}>
              {title}
            </span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[13px] italic text-faint">
            {by && <span>{by}</span>}
            <span className="num">p.{chunk.page}</span>
            {chunk.role === "distractor" && <Badge tone="warn">distractor doc</Badge>}
          </div>
        </div>
      </div>

      <Tip text={`Cosine similarity ${num(chunk.similarity, 3)} (1.0 = identical meaning)`}>
        <div className="flex w-full items-center gap-2">
          <div className="h-[3px] flex-1 overflow-hidden bg-ink/10">
            <motion.div
              className="h-full bg-accent"
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(0, Math.min(1, chunk.similarity)) * 100}%` }}
              transition={{ duration: 0.7, ease: "easeOut" }}
            />
          </div>
          <span className="num text-[12px] text-dim">{num(chunk.similarity, 2)}</span>
        </div>
      </Tip>

      <p className={clsx("text-[14px] leading-relaxed text-ink", !open && "line-clamp-4")}>{chunk.text}</p>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="focus-ring self-start text-[13.5px] text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
      >
        {open ? "Show less" : "Show more"}
      </button>
    </motion.li>
  );
}

export function ContextPanel({
  chunks,
  recall,
}: {
  chunks: ChunkOut[];
  recall?: number | null;
}) {
  return (
    <section className="section-rule" aria-labelledby="ctx-title">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="eyebrow mb-1">Shared retrieval</div>
          <h2 id="ctx-title" className="text-[21px] font-semibold">
            Retrieved context
          </h2>
          <p className="mt-0.5 text-[15px] text-dim">
            The same top-4 passages feed Baseline, Structured, RRR and Self-RAG. The control sees none.
          </p>
        </div>
        {recall !== undefined && recall !== null && (
          <Tip text="Share of the gold facts the judge found in these four passages. A ceiling on how well any strategy can cover the question.">
            <Badge tone={recall >= 0.99 ? "good" : recall >= 0.5 ? "neutral" : "warn"}>
              Context holds {Math.round(recall * 100)}% of gold facts
            </Badge>
          </Tip>
        )}
      </div>
      <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {chunks.map((c) => (
          <Passage key={c.chunk_id} chunk={c} />
        ))}
      </ul>
    </section>
  );
}
