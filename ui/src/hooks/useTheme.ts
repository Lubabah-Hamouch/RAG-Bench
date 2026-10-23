import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

const read = (): Theme =>
  document.documentElement.dataset.theme === "dark" ? "dark" : "light";

/** Theme preference only: it is the one thing this app persists, and it holds no user data. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const toggle = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === "dark" ? "light" : "dark";
      try {
        localStorage.setItem("ragbench-theme", next);
      } catch {
        /* storage unavailable: the choice just lasts for this session */
      }
      return next;
    });
  }, []);
  return [theme, toggle];
}
