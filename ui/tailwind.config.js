/** @type {import('tailwindcss').Config} */
const channel = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    // Lab-report look: square corners everywhere (0-2px), no shadows.
    borderRadius: {
      none: "0",
      sm: "1px",
      DEFAULT: "1px",
      md: "2px",
      lg: "2px",
      xl: "2px",
      "2xl": "2px",
      "3xl": "2px",
      full: "2px",
    },
    boxShadow: { none: "none", DEFAULT: "none", sm: "none", md: "none", lg: "none", card: "none", glow: "none" },
    extend: {
      borderColor: { DEFAULT: "rgb(var(--rule))", rule: "rgb(var(--rule))" },
      colors: {
        bg: channel("bg"),
        elev: channel("elev"),
        ink: channel("ink"),
        dim: channel("dim"),
        faint: channel("faint"),
        line: channel("line"),
        rule: channel("rule"),
        accent: channel("accent"),
        good: channel("good"),
        warn: channel("warn"),
        bad: channel("bad"),
        s: {
          baseline: channel("s-baseline"),
          structured: channel("s-structured"),
          rrr: channel("s-rrr"),
          self_rag: channel("s-self-rag"),
          no_retrieval: channel("s-control"),
        },
      },
      fontFamily: {
        serif: ['"Source Serif 4 Variable"', "Georgia", "Cambria", "serif"],
        sans: ['"Source Serif 4 Variable"', "Georgia", "Cambria", "serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: { shimmer: "shimmer 1.6s infinite" },
    },
  },
  plugins: [],
};
