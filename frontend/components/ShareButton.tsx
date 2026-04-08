"use client";

import { useState, useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";

interface ShareButtonProps {
  analysisId: string;
  score: number | null;
  type: string;
  lang: "en" | "zh";
  className?: string;
}

interface SocialPlatform {
  id: string;
  name: string;
  color: string;
  icon: React.ReactNode;
  getUrl?: (url: string, text: string) => string;
  copyOnly?: boolean;
}

function buildShareText(score: number | null, type: string, lang: "en" | "zh"): string {
  const tier = type.toUpperCase();
  const hasScore = typeof score === "number" && Number.isFinite(score);
  if (lang === "zh") {
    if (!hasScore) {
      return `我刚完成了 Stellar AI 高尔夫挥杆分析（${tier}），暂不可评分，查看完整报告：`;
    }
    return `我刚完成了 Stellar AI 高尔夫挥杆分析（${tier}），综合评分 ${score} 分，查看完整报告：`;
  }
  if (!hasScore) {
    return `I just got my golf swing analyzed by Stellar AI (${tier}) — score unavailable. Check the full report:`;
  }
  return `I just got my golf swing analyzed by Stellar AI (${tier}) — scored ${score}. Check the full report:`;
}

function computeSharePopoverLayout(trigger: DOMRectReadOnly) {
  const margin = 12;
  const vw = typeof window !== "undefined" ? window.innerWidth : 400;
  const vh = typeof window !== "undefined" ? window.innerHeight : 800;
  const width = Math.min(384, vw - margin * 2);
  let left = trigger.left + trigger.width / 2 - width / 2;
  left = Math.max(margin, Math.min(left, vw - width - margin));
  const estHeight = Math.min(vh * 0.88, 520);
  let top = trigger.bottom + margin;
  if (top + estHeight > vh - margin) {
    top = trigger.top - estHeight - margin;
  }
  if (top < margin) top = margin;
  const maxHeight = Math.max(200, vh - top - margin);
  return { top, left, width, maxHeight };
}

export default function ShareButton({ analysisId, score, type, lang, className }: ShareButtonProps) {
  const [loading, setLoading] = useState(false);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [popoverLayout, setPopoverLayout] = useState<{
    top: number;
    left: number;
    width: number;
    maxHeight: number;
  } | null>(null);

  useEffect(() => {
    function handleOutsideClick(e: MouseEvent) {
      const t = e.target as Node;
      if (modalRef.current?.contains(t)) return;
      if (triggerRef.current?.contains(t)) return;
      setShowModal(false);
    }
    if (showModal) document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [showModal]);

  useLayoutEffect(() => {
    if (!showModal || !shareUrl) {
      setPopoverLayout(null);
      return;
    }
    const update = () => {
      const btn = triggerRef.current;
      if (!btn) return;
      setPopoverLayout(computeSharePopoverLayout(btn.getBoundingClientRect()));
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [showModal, shareUrl]);

  const generateShare = useCallback(async () => {
    if (shareUrl) {
      setShowModal(true);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const token =
        typeof window !== "undefined" ? localStorage.getItem("stellar_token") || "" : "";
      const res = await fetch("/api/share", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ analysis_id: analysisId }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError((data as { detail?: string }).detail || (lang === "zh" ? "生成分享链接失败" : "Failed to generate share link"));
        return;
      }
      const data = (await res.json()) as { url: string };
      setShareUrl(data.url);
      setShowModal(true);
    } catch {
      setError(lang === "zh" ? "网络错误，请重试" : "Network error, please retry");
    } finally {
      setLoading(false);
    }
  }, [analysisId, shareUrl, lang]);

  const handleCopy = useCallback(async () => {
    if (!shareUrl) return;
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const el = document.createElement("textarea");
      el.value = shareUrl;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [shareUrl]);

  const shareText = buildShareText(score, type, lang);

  const platforms: SocialPlatform[] = [
    // ── China ──
    {
      id: "weibo",
      name: "微博",
      color: "#e6162d",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M10.098 20.323c-3.977.391-7.414-1.406-7.672-4.02-.259-2.609 2.759-5.047 6.74-5.441 3.979-.394 7.413 1.404 7.671 4.018.259 2.614-2.759 5.049-6.74 5.443zm5.44-11.93c-.412-.118-.697-.202-.482-.728.465-1.17.513-2.176.013-2.898-.96-1.39-3.585-1.315-6.601.034 0 0-.942.412-.7-.336.463-1.489.393-2.736-.328-3.456-1.641-1.642-6.003.064-9.733 3.797C-5.425 7.909-7 11.148-7 14.075c0 5.667 7.279 9.104 14.415 9.104 9.334 0 15.55-5.426 15.55-9.739 0-2.607-2.202-4.088-3.63-4.67-.413-.17-1.027-.388-1.797-.377z"/>
        </svg>
      ),
      getUrl: (url, text) =>
        `https://service.weibo.com/share/share.php?url=${encodeURIComponent(url)}&title=${encodeURIComponent(text + " " + url)}`,
    },
    {
      id: "qq",
      name: "QQ空间",
      color: "#12b7f5",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M12.003 2c-4.418 0-8 3.582-8 8 0 2.123.822 4.056 2.163 5.508L4.5 21l5.18-2.347C10.53 18.89 11.25 19 12.003 19c4.418 0 8-3.582 8-8s-3.582-9-8-9zm0 14c-.67 0-1.32-.09-1.94-.258l-.139-.04-3.071 1.393 1.013-2.98-.204-.27A5.978 5.978 0 016.003 10c0-3.309 2.691-6 6-6s6 2.691 6 6-2.691 6-6 6z"/>
        </svg>
      ),
      getUrl: (url, text) =>
        `https://sns.qzone.qq.com/cgi-bin/qzshare/cgi_qzshare_onekey?url=${encodeURIComponent(url)}&title=${encodeURIComponent(text)}&summary=${encodeURIComponent(text)}`,
    },
    {
      id: "wechat",
      name: lang === "zh" ? "微信" : "WeChat",
      color: "#07c160",
      copyOnly: true,
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M8.691 2.188C3.891 2.188 0 5.476 0 9.53c0 2.212 1.17 4.203 3.002 5.55a.59.59 0 0 1 .213.665l-.39 1.48c-.019.07-.048.141-.048.213 0 .163.13.295.29.295a.326.326 0 0 0 .167-.054l1.903-1.114a.864.864 0 0 1 .717-.098 10.16 10.16 0 0 0 2.837.403c.276 0 .543-.027.811-.05-.857-2.578.325-5.44 3.108-7.087C13.562 8.22 15.376 7.75 17 7.75c.065 0 .13.003.195.005C16.343 4.56 12.815 2.19 8.691 2.188zm-2.45 3.61c.576 0 1.043.466 1.043 1.043a1.043 1.043 0 0 1-2.086 0c0-.577.467-1.043 1.043-1.043zm4.9 0c.576 0 1.043.466 1.043 1.043a1.043 1.043 0 0 1-2.086 0c0-.577.467-1.043 1.043-1.043zm3.83 3.55c-4.124 0-7.462 2.938-7.462 6.563 0 3.624 3.338 6.563 7.462 6.563.865 0 1.69-.124 2.465-.35a.61.61 0 0 1 .525.071l1.553.908a.289.289 0 0 0 .136.046.24.24 0 0 0 .237-.242c0-.059-.021-.115-.039-.175l-.318-1.207a.483.483 0 0 1 .175-.544C21.101 19.688 22 18.054 22 16.25c0-3.625-3.338-6.563-7.462-6.563h-.576zm-2.44 3.04c.47 0 .852.38.852.852a.852.852 0 0 1-1.704 0c0-.472.382-.852.852-.852zm4.892 0c.47 0 .852.38.852.852a.852.852 0 0 1-1.704 0c0-.472.382-.852.852-.852z"/>
        </svg>
      ),
    },
    // ── International ──
    {
      id: "twitter",
      name: "X / Twitter",
      color: "#000000",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.744l7.73-8.835L1.254 2.25H8.08l4.253 5.622zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
        </svg>
      ),
      getUrl: (url, text) =>
        `https://twitter.com/intent/tweet?text=${encodeURIComponent(text + " " + url)}`,
    },
    {
      id: "facebook",
      name: "Facebook",
      color: "#1877f2",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
        </svg>
      ),
      getUrl: (url) =>
        `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
    },
    {
      id: "whatsapp",
      name: "WhatsApp",
      color: "#25d366",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/>
        </svg>
      ),
      getUrl: (url, text) =>
        `https://wa.me/?text=${encodeURIComponent(text + " " + url)}`,
    },
    {
      id: "telegram",
      name: "Telegram",
      color: "#2ca5e0",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/>
        </svg>
      ),
      getUrl: (url, text) =>
        `https://t.me/share/url?url=${encodeURIComponent(url)}&text=${encodeURIComponent(text)}`,
    },
    {
      id: "line",
      name: "Line",
      color: "#00b900",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
          <path d="M19.365 9.863c.349 0 .63.285.63.631 0 .345-.281.63-.63.63H17.61v1.125h1.755c.349 0 .63.283.63.63 0 .344-.281.629-.63.629h-2.386c-.345 0-.627-.285-.627-.629V8.108c0-.345.282-.63.63-.63h2.386c.346 0 .627.285.627.63 0 .349-.281.63-.63.63H17.61v1.125h1.755zm-3.855 3.016c0 .27-.174.51-.432.596-.064.021-.133.031-.199.031-.211 0-.391-.09-.51-.25l-2.443-3.317v2.94c0 .344-.279.629-.631.629-.346 0-.626-.285-.626-.629V8.108c0-.27.173-.51.43-.595.06-.023.136-.033.194-.033.195 0 .375.104.495.254l2.462 3.33V8.108c0-.345.282-.63.63-.63.345 0 .63.285.63.63v4.771zm-5.741 0c0 .344-.282.629-.631.629-.345 0-.627-.285-.627-.629V8.108c0-.345.282-.63.63-.63.346 0 .628.285.628.63v4.771zm-2.466.629H4.917c-.345 0-.63-.285-.63-.629V8.108c0-.345.285-.63.63-.63.348 0 .63.285.63.63v4.141h1.756c.348 0 .629.283.629.63 0 .344-.282.629-.629.629M24 10.314C24 4.943 18.615.572 12 .572S0 4.943 0 10.314c0 4.811 4.27 8.842 10.035 9.608.391.082.923.258 1.058.59.12.301.079.766.038 1.08l-.164 1.02c-.045.301-.24 1.186 1.049.645 1.291-.539 6.916-4.078 9.436-6.975C23.176 14.393 24 12.458 24 10.314"/>
        </svg>
      ),
      getUrl: (url) =>
        `https://social-plugins.line.me/lineit/share?url=${encodeURIComponent(url)}`,
    },
  ];

  return (
    <>
      {/* Share trigger button */}
      <button
        ref={triggerRef}
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          generateShare();
        }}
        disabled={loading}
        title={lang === "zh" ? "分享此分析" : "Share this analysis"}
        className={`flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-xs text-white/50 transition hover:border-white/20 hover:bg-white/[0.08] hover:text-white/70 disabled:opacity-50 ${className ?? ""}`}
      >
        {loading ? (
          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/60" />
        ) : (
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M7.217 10.907a2.25 2.25 0 1 0 0 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186 9.566-5.314m-9.566 7.5 9.566 5.314m0 0a2.25 2.25 0 1 0 3.935 2.186 2.25 2.25 0 0 0-3.935-2.186zm0-12.814a2.25 2.25 0 1 0 3.933-2.185 2.25 2.25 0 0 0-3.933 2.185z" />
          </svg>
        )}
        {lang === "zh" ? "分享" : "Share"}
      </button>

      {error && (
        <p className="mt-1 text-[11px] text-red-400/70">{error}</p>
      )}

      {/* Share modal — portal + anchor near trigger so history card backdrop-filter does not trap fixed */}
      {showModal && shareUrl && popoverLayout && typeof document !== "undefined" && document.body
        ? createPortal(
            <>
              <div
                className="fixed inset-0 z-[210] bg-black/60"
                aria-hidden
                onClick={() => setShowModal(false)}
              />
              <div
                ref={modalRef}
                className="fixed z-[211] rounded-2xl border border-white/10 bg-[#0f0a1e] p-5 shadow-2xl"
                style={{
                  top: popoverLayout.top,
                  left: popoverLayout.left,
                  width: popoverLayout.width,
                  maxHeight: popoverLayout.maxHeight,
                  overflowY: "auto",
                }}
                onClick={(e) => e.stopPropagation()}
              >
            {/* Header */}
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">
                {lang === "zh" ? "分享分析报告" : "Share Analysis Report"}
              </h3>
              <button
                onClick={() => setShowModal(false)}
                className="flex h-7 w-7 items-center justify-center rounded-full text-white/40 transition hover:bg-white/10 hover:text-white/70"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Link copy area */}
            <div className="mb-4 flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2.5">
              <span className="flex-1 truncate text-xs text-white/50">{shareUrl}</span>
              <button
                onClick={handleCopy}
                className="flex-shrink-0 rounded-lg bg-brand-purple/80 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-brand-purple"
              >
                {copied ? (lang === "zh" ? "✓ 已复制" : "✓ Copied") : (lang === "zh" ? "复制" : "Copy")}
              </button>
            </div>

            {/* Platform note */}
            <p className="mb-3 text-[11px] text-white/30">
              {lang === "zh" ? "任何人打开链接都可查看完整报告，无需登录" : "Anyone with this link can view the full report without logging in"}
            </p>

            {/* Social media grid */}
            <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-white/30">
              {lang === "zh" ? "分享到" : "Share to"}
            </p>
            <div className="grid grid-cols-4 gap-3">
              {platforms.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    if (p.copyOnly) {
                      handleCopy();
                      return;
                    }
                    if (p.getUrl) {
                      window.open(p.getUrl(shareUrl, shareText), "_blank", "noopener,noreferrer");
                    }
                  }}
                  className="flex flex-col items-center gap-1.5 rounded-xl p-2 transition hover:bg-white/[0.06]"
                  title={p.name}
                >
                  <span
                    className="flex h-9 w-9 items-center justify-center rounded-full text-white"
                    style={{ backgroundColor: p.color + "22", color: p.color }}
                  >
                    {p.icon}
                  </span>
                  <span className="text-[10px] leading-tight text-white/40">{p.name}</span>
                  {p.copyOnly && (
                    <span className="text-[9px] text-white/25">{lang === "zh" ? "复制链接" : "Copy link"}</span>
                  )}
                </button>
              ))}
            </div>

            <p className="mt-4 text-[10px] text-white/20">
              {lang === "zh"
                ? "⚠ 分享链接公开可见，请谨慎分享个人数据"
                : "⚠ Share link is publicly accessible. Share with care."}
            </p>
              </div>
            </>,
            document.body,
          )
        : null}
    </>
  );
}
