"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  image: string;
  source: string;
  category: string;
  published_at: string;
  url: string;
}

export default function NewsCarousel() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const FALLBACK_IMAGE = "/logo.svg";

  const fetchNews = useCallback(async () => {
    try {
      // Use local API proxy and bypass stale browser caches.
      const res = await fetch(`/api/news?limit=10&_=${Date.now()}`, { cache: "no-store" });
      if (res.ok) {
        const data = await res.json();
        const normalized = (data.news || []).map((n: NewsItem) => ({
          ...n,
          image: n.image || FALLBACK_IMAGE,
        }));
        setNews(normalized);
      }
    } catch {
      setNews([
        { id: "f1", title: "Master Your Grip: The Foundation of Every Great Swing", summary: "", image: "", source: "Stellar AI", category: "Tips", published_at: new Date().toISOString(), url: "#" },
        { id: "f2", title: "PGA Tour 2026: Season Highlights and Key Takeaways", summary: "", image: "", source: "Golf Digest", category: "Tour", published_at: new Date().toISOString(), url: "#" },
        { id: "f3", title: "How AI is Revolutionizing Golf Coaching", summary: "", image: "", source: "Stellar AI", category: "Tech", published_at: new Date().toISOString(), url: "#" },
        { id: "f4", title: "5 Drills to Improve Your Weight Transfer", summary: "", image: "", source: "Stellar AI", category: "Training", published_at: new Date().toISOString(), url: "#" },
      ]);
    } finally {
      setLoading(false);
    }
  }, [FALLBACK_IMAGE]);

  useEffect(() => {
    fetchNews();
    const refresh = setInterval(fetchNews, 300000); // refresh every 5 minutes
    return () => clearInterval(refresh);
  }, [fetchNews]);

  const scrollToIndex = useCallback(
    (index: number) => {
      if (!scrollRef.current || news.length === 0) return;
      const safeIndex = ((index % news.length) + news.length) % news.length;
      setCurrentIndex(safeIndex);
      const card = scrollRef.current.children[safeIndex] as HTMLElement;
      if (card) {
        // Scroll only within the carousel container, not the page
        scrollRef.current.scrollTo({
          left: card.offsetLeft - scrollRef.current.offsetLeft,
          behavior: "smooth",
        });
      }
    },
    [news.length]
  );

  useEffect(() => {
    if (news.length === 0) return;

    timerRef.current = setInterval(() => {
      scrollToIndex(currentIndex + 1);
    }, 3000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [currentIndex, news.length, scrollToIndex]);

  function handleTouchStart() {
    if (timerRef.current) clearInterval(timerRef.current);
  }

  function handleTouchEnd() {
    timerRef.current = setInterval(() => {
      scrollToIndex(currentIndex + 1);
    }, 3000);
  }

  if (loading) {
    return (
      <section className="mx-auto max-w-7xl px-6 py-12">
        <div className="flex gap-4 overflow-hidden">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="h-64 w-72 flex-shrink-0 animate-pulse rounded-xl bg-white/5"
            />
          ))}
        </div>
      </section>
    );
  }

  if (news.length === 0) return null;

  return (
    <section className="mx-auto max-w-7xl px-6 py-12">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">
          Golf News <span className="text-white/40 text-lg font-normal">/ 高尔夫新闻</span>
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => scrollToIndex(currentIndex - 1)}
            className="flex h-8 w-8 items-center justify-center rounded-full border border-white/10 text-white/50 transition hover:border-brand-purple/30 hover:text-brand-gold"
          >
            ‹
          </button>
          <button
            onClick={() => scrollToIndex(currentIndex + 1)}
            className="flex h-8 w-8 items-center justify-center rounded-full border border-white/10 text-white/50 transition hover:border-brand-purple/30 hover:text-brand-gold"
          >
            ›
          </button>
        </div>
      </div>

      <div
        ref={scrollRef}
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
        className="flex gap-4 overflow-x-auto scroll-smooth pb-4 snap-x snap-mandatory"
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        {news.map((item, i) => (
          <a
            key={item.id}
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className={`group flex-shrink-0 snap-center w-72 overflow-hidden rounded-xl border transition-all duration-300 ${
              i === currentIndex
                ? "border-brand-purple/30 shadow-lg shadow-brand-purple/10"
                : "border-white/5 hover:border-white/20"
            }`}
          >
            <div className="relative h-40 overflow-hidden bg-white/5">
              <img
                src={item.image}
                alt={item.title}
                className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-110"
                loading="lazy"
                onError={(e) => {
                  const img = e.currentTarget;
                  if (img.src !== FALLBACK_IMAGE) img.src = FALLBACK_IMAGE;
                }}
              />
              <div className="absolute left-2 top-2">
                <span className="rounded-full bg-brand-purple/80 px-2 py-0.5 text-[10px] font-semibold text-white">
                  {item.category}
                </span>
              </div>
            </div>
            <div className="bg-brand-card/80 p-4">
              <h3 className="mb-2 line-clamp-2 text-sm font-semibold text-white group-hover:text-brand-gold transition-colors">
                {item.title}
              </h3>
              <div className="flex items-center justify-between text-xs text-white/40">
                <span>{item.source}</span>
                <span>
                  {(() => {
                    const d = new Date(item.published_at);
                    return isNaN(d.getTime()) ? item.published_at : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
                  })()}
                </span>
              </div>
            </div>
          </a>
        ))}
      </div>

      {/* Dots */}
      <div className="mt-4 flex justify-center gap-1.5">
        {news.map((_, i) => (
          <button
            key={i}
            onClick={() => scrollToIndex(i)}
            className={`h-1.5 rounded-full transition-all ${
              i === currentIndex
                ? "w-6 bg-brand-purple"
                : "w-1.5 bg-white/20 hover:bg-white/40"
            }`}
          />
        ))}
      </div>
    </section>
  );
}
