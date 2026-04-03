/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "images.unsplash.com" },
      { protocol: "https", hostname: "*.r2.cloudflarestorage.com" },
    ],
  },
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000",
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
    // MediaPipe R2 URL is public in the browser anyway. Some hosts only allow "Secret" env names
    // without NEXT_PUBLIC_* — map those here so `next build` can still inline the client bundle.
    NEXT_PUBLIC_MEDIAPIPE_CDN_BASE: (
      process.env.NEXT_PUBLIC_MEDIAPIPE_CDN_BASE ||
      process.env.MEDIAPIPE_CDN_BASE ||
      ""
    ).trim(),
    NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK: (
      process.env.NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK ||
      process.env.MEDIAPIPE_ALLOW_FOREIGN_FALLBACK ||
      ""
    ).trim(),
  },
  webpack: (config) => {
    config.resolve.alias["@mediapipe/tasks-vision"] = false;
    return config;
  },
};

module.exports = nextConfig;
