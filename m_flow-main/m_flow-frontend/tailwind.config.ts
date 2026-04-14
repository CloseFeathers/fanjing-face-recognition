import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // x.ai inspired high-end gray palette
        "xai-black": "#0a0a0a",
        "xai-dark": "#111111",
        "xai-surface": "#1a1a1a",
        "xai-card": "#222222",
        "xai-border": "#2a2a2a",
        "xai-muted": "#3a3a3a",
        "xai-gray": "#666666",
        "xai-text": "#a0a0a0",
        "xai-light": "#e0e0e0",
        "xai-white": "#f5f5f5",
        "xai-accent": "#4a9eff",
        "xai-accent-hover": "#3d8ce8",
        "xai-success": "#22c55e",
        "xai-warning": "#eab308",
        "xai-error": "#ef4444",

        // Desaturated overrides — all Tailwind colors shifted to luxury grey
        blue: {
          300: "#7da3c7",
          400: "#6b92b8",
          500: "#5880a5",
          600: "#486d8e",
          700: "#3a5870",
          800: "#2d4354",
          900: "#1f2f3b",
          950: "#151f28",
        },
        amber: {
          300: "#d4bd7a",
          400: "#c4ab65",
          500: "#ad9652",
          600: "#8f7b43",
          700: "#6e5f34",
          800: "#4e4326",
          900: "#332c1a",
          950: "#221d12",
        },
        emerald: {
          400: "#5db890",
          500: "#4da37d",
          600: "#3f8768",
          700: "#326b53",
          800: "#264f3e",
          900: "#1b382c",
          950: "#12251d",
        },
        green: {
          400: "#5db88a",
          500: "#4da378",
          600: "#3f8763",
          700: "#326b4f",
          800: "#264f3b",
          900: "#1b382a",
          950: "#12251c",
        },
        purple: {
          400: "#9f8ec5",
          500: "#8b7ab2",
          600: "#736496",
          700: "#5b4f77",
          800: "#443b59",
          900: "#2f2a3e",
          950: "#201c2a",
        },
        red: {
          400: "#c47070",
          500: "#b05a5a",
          600: "#944a4a",
          700: "#753a3a",
          800: "#562c2c",
          900: "#3b1f1f",
          950: "#281515",
        },
        cyan: {
          600: "#3f8a82",
        },
        rose: {
          400: "#c48080",
          500: "#b06a6a",
        },
        yellow: {
          400: "#d4ba6a",
          500: "#b8a255",
          600: "#968543",
        },
        orange: {
          400: "#c49a65",
          500: "#ad8552",
        },
      },
      fontFamily: {
        sans: ["SF Pro Display", "Inter", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["SF Mono", "JetBrains Mono", "Fira Code", "monospace"],
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-up": "slideUp 0.3s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
