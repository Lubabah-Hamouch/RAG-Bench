import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

/** Which retrieved passage a hovered claim points at, shared between the cards and the context. */
interface Highlight {
  passage: number | null;
  strategy: string | null;
  set: (strategy: string | null, passage: number | null) => void;
}

const Ctx = createContext<Highlight>({ passage: null, strategy: null, set: () => {} });

export function HighlightProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<{ passage: number | null; strategy: string | null }>({
    passage: null,
    strategy: null,
  });
  const value = useMemo<Highlight>(
    () => ({ ...state, set: (strategy, passage) => setState({ strategy, passage }) }),
    [state],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useHighlight = (): Highlight => useContext(Ctx);
