import { useCallback, useEffect, useState } from "react";

export type View = "explore" | "benchmark" | "method";

export interface Route {
  view: View;
  qid: string | null;
}

/** Hash routes, so the app works on any static host: #/explore/Q05, #/benchmark, #/method */
export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const head = parts[0];
  if (head === "benchmark") return { view: "benchmark", qid: null };
  if (head === "method") return { view: "method", qid: null };
  return { view: "explore", qid: head === "explore" ? (parts[1] ?? null) : null };
}

export function toHash(route: Route): string {
  if (route.view === "explore") return route.qid ? `#/explore/${route.qid}` : "#/explore";
  return `#/${route.view}`;
}

export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((r: Route) => {
    const hash = toHash(r);
    if (window.location.hash !== hash) window.location.hash = hash;
    else setRoute(r);
  }, []);
  return [route, navigate];
}
