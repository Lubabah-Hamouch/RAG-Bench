import { describe, expect, it } from "vitest";
import indexHtml from "../../index.html?raw";
import blueprint from "../../../render.yaml?raw";

// The static site's Content-Security-Policy lives in render.yaml. These checks fail when it falls
// out of step with index.html (an edited inline script) or with the API URL the build is given.
const csp = blueprint.match(/name: Content-Security-Policy\s+value: "([^"]+)"/)![1];
const directives: Record<string, string[]> = Object.fromEntries(
  csp.split(";").map((d) => {
    const [name, ...sources] = d.trim().split(/\s+/);
    return [name, sources];
  }),
);
const apiUrl = blueprint.match(/key: VITE_API_URL\s+value: (\S+)/)![1];

/** The CSP source for an inline script: base64 of the SHA-256 of its exact text. */
async function scriptHash(body: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(body));
  return `'sha256-${btoa(String.fromCharCode(...new Uint8Array(digest)))}'`;
}

describe("static site CSP", () => {
  it("allows exactly the inline scripts in index.html, by hash", async () => {
    const bodies = [...indexHtml.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
    expect(bodies.length).toBeGreaterThan(0);
    const hashes = await Promise.all(bodies.map(scriptHash));
    expect(directives["script-src"]).toEqual(["'self'", ...hashes]);
  });

  it("lets the page reach the API it was built against, and nothing else", () => {
    expect(directives["connect-src"]).toEqual(["'self'", new URL(apiUrl).origin]);
  });

  it("is as strict as the policy the API used to send", () => {
    expect(directives["default-src"]).toEqual(["'self'"]);
    expect(directives["object-src"]).toEqual(["'none'"]);
    expect(directives["frame-ancestors"]).toEqual(["'none'"]);
    expect(directives["base-uri"]).toEqual(["'self'"]);
    expect(directives["form-action"]).toEqual(["'self'"]);
    expect(directives["img-src"]).toEqual(["'self'", "data:"]);
    expect(directives["font-src"]).toEqual(["'self'", "data:"]);
    expect(directives["style-src"]).toEqual(["'self'", "'unsafe-inline'"]);
  });
});
