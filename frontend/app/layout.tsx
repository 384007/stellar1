import type { Metadata, Viewport } from "next";
import { Bebas_Neue, Rajdhani } from "next/font/google";
import "./globals.css";
import SwRegister from "./sw-register";

const bebasNeue = Bebas_Neue({
  weight: "400",
  subsets: ["latin"],
  display: "swap",
  variable: "--font-bebas",
});

const rajdhani = Rajdhani({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  display: "swap",
  variable: "--font-rajdhani",
});

const ICON_VERSION = "25";

export const viewport: Viewport = {
  themeColor: "#7c3aed",
  width: "device-width",
  initialScale: 1,
};

export const metadata: Metadata = {
  title: "Stellar - 智能高尔夫挥杆分析",
  description:
    "AI驱动的专业高尔夫挥杆分析平台。即时获取握杆、站姿、后摆、下杆、收杆的专业反馈。Powered by Stellar AI.",
  keywords: ["golf", "高尔夫", "swing analysis", "AI", "biomechanics", "stellar ai"],
  manifest: `/manifest.json?v=${ICON_VERSION}`,
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Stellar",
  },
  icons: {
    apple: [{ url: `/icon-192.png?v=${ICON_VERSION}`, sizes: "192x192" }],
    icon: [{ url: "/logo.svg", type: "image/svg+xml" }],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" className={`dark ${bebasNeue.variable} ${rajdhani.variable}`}>
      <head>
        <link rel="icon" type="image/svg+xml" href="/logo.svg" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-touch-fullscreen" content="yes" />
        <link rel="apple-touch-icon" href={`/icon-192.png?v=${ICON_VERSION}`} />
        <link rel="apple-touch-startup-image" href={`/icon-512.png?v=${ICON_VERSION}`} />
      </head>
      <body className="min-h-screen antialiased">
        <div className="relative z-10">{children}</div>
        <SwRegister />
      </body>
    </html>
  );
}
