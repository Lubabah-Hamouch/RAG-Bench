import clsx from "clsx";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect, useId, useState, type ReactNode } from "react";

export function Logo({ size = 28 }: { size?: number }) {
  const dots: [number, number, string][] = [
    [6, 22, "--s-baseline"],
    [11, 12.6, "--s-structured"],
    [16, 9.5, "--s-rrr"],
    [21, 12.6, "--s-self-rag"],
    [26, 22, "--s-control"],
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <path d="M6 22 Q16 4 26 22" fill="none" stroke="rgb(var(--ink) / 0.35)" strokeWidth="1.2" />
      {dots.map(([cx, cy, v]) => (
        <rect key={v} x={cx - 2.6} y={cy - 2.6} width="5.2" height="5.2" fill={`rgb(var(${v}))`} />
      ))}
    </svg>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className,
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
  className?: string;
  title?: string;
}) {
  const tones = {
    neutral: "text-dim border-rule",
    good: "text-good border-good/50",
    warn: "text-warn border-warn/50",
    bad: "text-bad border-bad/50",
    accent: "text-accent border-accent/50",
  } as const;
  return (
    <span
      title={title}
      className={clsx(
        "inline-flex items-center gap-1 border px-1.5 py-0.5 text-[12px] leading-4",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Chip({
  icon,
  label,
  value,
  title,
}: {
  icon?: ReactNode;
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1.5 border border-rule px-1.5 py-0.5 text-[12px] text-dim"
    >
      {icon}
      <span className="text-faint">{label}</span>
      <span className="num text-ink/90">{value}</span>
    </span>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  id,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  id?: string;
}) {
  const auto = useId();
  const sid = id ?? auto;
  return (
    <button
      id={sid}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={clsx(
        "focus-ring relative h-6 w-11 shrink-0 border transition-colors duration-150",
        checked ? "border-accent bg-accent" : "hairline bg-ink/10",
      )}
    >
      <motion.span
        layout
        transition={{ type: "spring", stiffness: 500, damping: 32 }}
        className="absolute top-0.5 h-[18px] w-[18px] bg-bg"
        style={{ left: checked ? 22 : 2 }}
      />
    </button>
  );
}

/** A small hover/focus tooltip that needs no portal. */
export function Tip({ text, children }: { text: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open && (
        <span
          role="tooltip"
          className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 w-max max-w-[240px] -translate-x-1/2 border bg-elev px-2.5 py-1.5 text-[12.5px] leading-snug text-ink hairline"
        >
          {text}
        </span>
      )}
    </span>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={clsx("animate-pulse bg-ink/[0.07]", className)} />
  );
}

/** Count a number up (or down) when it changes. */
export function useAnimatedNumber(value: number | null, duration = 0.7): number | null {
  const mv = useMotionValue(value ?? 0);
  const rounded = useTransform(mv, (v) => v);
  const [display, setDisplay] = useState<number | null>(value);
  useEffect(() => {
    if (value === null) {
      setDisplay(null);
      return;
    }
    // A real score must never display as "-": when the value arrives after an unscored render the
    // animation may not move at all (0 -> 0 fires no change), so show the current value at once
    // and pin the exact target when the animation ends.
    setDisplay((d) => d ?? mv.get());
    const controls = animate(mv, value, {
      duration,
      ease: "easeOut",
      onComplete: () => setDisplay(value),
    });
    const unsub = rounded.on("change", (v) => setDisplay(v));
    return () => {
      controls.stop();
      unsub();
    };
  }, [value, duration, mv, rounded]);
  return display;
}

/** Flat bar meter for a 0-1 score: a mono readout over a hairline track. */
export function Meter({
  label,
  value,
  color,
  hint,
}: {
  label: string;
  value: number | null | undefined;
  color: string;
  size?: number;
  hint?: string;
}) {
  const v = value ?? null;
  const shown = useAnimatedNumber(v);
  const frac = v === null ? 0 : Math.max(0, Math.min(1, v));
  return (
    <Tip text={hint ?? label}>
      <div className="min-w-[4.5rem] flex-1" aria-label={`${label} ${v === null ? "not available" : v.toFixed(2)}`}>
        <div className="text-[12.5px] italic text-faint">{label}</div>
        <div className="num text-[22px] leading-tight text-ink" data-testid="meter-value">
          {shown === null ? <span className="text-faint">–</span> : shown.toFixed(2)}
        </div>
        <div className="mt-1 h-[3px] w-full bg-ink/10">
          <motion.div
            className="h-full"
            style={{ background: color }}
            initial={{ width: 0 }}
            animate={{ width: `${frac * 100}%` }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          />
        </div>
      </div>
    </Tip>
  );
}

export function SectionTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div>
        {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
        <h2 className="text-[21px] font-semibold leading-snug">{title}</h2>
      </div>
      {children}
    </div>
  );
}

/** Compact stat row: a mono figure under an italic label, separated by hairline rules. */
export function StatRow({ items }: { items: { value: string; label: string }[] }) {
  return (
    <dl className="grid grid-cols-2 border-y hairline sm:flex sm:flex-wrap">
      {items.map((s, i) => (
        <div
          key={s.label}
          className={clsx(
            "px-4 py-2.5",
            i % 2 === 0 ? "pl-0 sm:pl-4" : "border-l hairline",
            i === 0 && "sm:pl-0",
            i >= 2 && "border-t hairline sm:border-t-0",
            i > 0 && "sm:border-l hairline",
          )}
        >
          <dt className="text-[13px] italic text-faint">{s.label}</dt>
          <dd className="num text-[22px] leading-tight text-ink">{s.value}</dd>
        </div>
      ))}
    </dl>
  );
}
