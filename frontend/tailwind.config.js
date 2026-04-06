/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          purple:       "#7c3aed",
          "purple-glow":"#9f5fff",
          gold:         "#f5c518",
          "gold-light": "#ffd85e",
          "gold-deep":  "#c9960a",
          dark:         "#0d0a1a",
          card:         "#1a1530",
          surface:      "#231d42",
          text:         "#f0eaff",
          muted:        "#8b7db5",
          // legacy aliases so old code still works
          green:        "#7c3aed",
          cyan:         "#9f5fff",
        },
      },
      fontFamily: {
        display: ["var(--font-bebas)", "'Bebas Neue'", "system-ui", "sans-serif"],
        body:    ["var(--font-rajdhani)", "'Rajdhani'", "system-ui", "sans-serif"],
      },
      animation: {
        "ticker-scroll": "ticker 35s linear infinite",
        "pulse-glow":    "pulseGlow 2s ease-in-out infinite",
        "fade-in":       "fadeIn 0.5s ease-out",
        "slide-up":      "slideUp 0.6s ease-out",
        "flow-line":     "flowLine 2s linear infinite",
        "ripple":        "ripple 1.5s ease-out infinite",
        "breathe":       "breathe 2s ease-in-out infinite",
        "shimmer":       "shimmer 3s linear infinite",
        "float":         "particleFloat 8s ease-in-out infinite alternate",
        "glow-pulse":    "glowPulse 2s ease-in-out infinite",
        /** Prov3 画笔色板：同锚点叠层淡入 */
        "color-popover": "colorPopover 0.26s cubic-bezier(0.22, 1, 0.36, 1) both",
      },
      keyframes: {
        ticker: {
          "0%":   { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 20px rgba(124,58,237,0.4)" },
          "50%":      { boxShadow: "0 0 40px rgba(245,197,24,0.6)" },
        },
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%":   { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        flowLine: {
          "0%":   { strokeDashoffset: "100" },
          "100%": { strokeDashoffset: "0" },
        },
        ripple: {
          "0%":   { transform: "scale(0.8)", opacity: "1" },
          "100%": { transform: "scale(2.5)", opacity: "0" },
        },
        breathe: {
          "0%, 100%": { transform: "scale(1)" },
          "50%":      { transform: "scale(1.2)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% center" },
          "100%": { backgroundPosition: "200% center" },
        },
        particleFloat: {
          "0%":   { transform: "translateY(0px)" },
          "100%": { transform: "translateY(-14px)" },
        },
        glowPulse: {
          "0%, 100%": { filter: "drop-shadow(0 0 6px rgba(245,197,24,0.6))" },
          "50%":      { filter: "drop-shadow(0 0 16px rgba(245,197,24,1))" },
        },
        colorPopover: {
          "0%": { opacity: "0", transform: "translate(-50%, -50%) scale(0.9)" },
          "100%": { opacity: "1", transform: "translate(-50%, -50%) scale(1)" },
        },
      },
      boxShadow: {
        "gold-glow":   "0 0 20px rgba(245,197,24,0.5), 0 0 40px rgba(245,197,24,0.2)",
        "purple-glow": "0 0 20px rgba(124,58,237,0.5), 0 0 40px rgba(124,58,237,0.2)",
        "card":        "0 4px 30px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04)",
      },
    },
  },
  plugins: [],
};
