import clsx from "clsx";
import { Check, X } from "lucide-react";
import { color, STRATEGY_ORDER, strategyShort } from "@/lib/strategies";
import type { AnswerOut } from "@/lib/types";
import { Tip } from "./ui";

/** Which gold facts each strategy's answer stated, fact by fact. */
export function CoverageMatrix({
  facts,
  answers,
}: {
  facts: string[];
  answers: Record<string, AnswerOut>;
}) {
  if (!facts.length) return null;
  const present = STRATEGY_ORDER.filter((s) => answers[s]);
  return (
    <section className="section-rule" aria-labelledby="cov-title">
      <div className="eyebrow mb-1">Completeness</div>
      <h2 id="cov-title" className="mb-3 text-[21px] font-semibold">
        Which key facts did each answer state?
      </h2>
      <div className="overflow-x-auto">
        <table className="ledger min-w-[560px]">
          <thead>
            <tr>
              <th className="w-full">Gold fact</th>
              {present.map((s) => (
                <th key={s} className="text-center">
                  <span className="inline-flex items-center gap-1.5 whitespace-nowrap not-italic">
                    <span className="h-2 w-2" style={{ background: color(s) }} />
                    {strategyShort(s)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {facts.map((fact, i) => (
              <tr key={i}>
                <td className="text-[15px] leading-snug">
                  <span className="num mr-2 text-faint">{i + 1}</span>
                  {fact}
                </td>
                {present.map((s) => {
                  const covered = answers[s]?.metrics.coverage?.covered?.[i];
                  const judged = answers[s]?.metrics.judged && covered !== undefined;
                  return (
                    <td key={s} className="text-center">
                      {!judged ? (
                        <span className="text-faint">–</span>
                      ) : (
                        <Tip text={covered ? "Stated in the answer" : "Missing from the answer"}>
                          <span
                            className={clsx(
                              "grid h-6 w-6 place-items-center",
                              covered ? "bg-good text-bg" : "border border-rule text-faint",
                            )}
                          >
                            {covered ? <Check size={13} strokeWidth={3} /> : <X size={12} strokeWidth={2.5} />}
                          </span>
                        </Tip>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
