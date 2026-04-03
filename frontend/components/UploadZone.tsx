"use client";

import { useState, useRef, useCallback } from "react";
import { compressVideoForUpload } from "@/lib/video-compress";

interface UploadZoneProps {
  onUploadComplete: (file: File) => void;
  lang: "en" | "zh";
  isPro?: boolean;
}

export default function UploadZone({ onUploadComplete, lang, isPro }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStage, setUploadStage] = useState("");
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const MAX_SIZE = 100 * 1024 * 1024;

  const handleFile = useCallback(
    async (file: File) => {
      setError("");

      if (!file.type.includes("video/")) {
        setError(lang === "en" ? "Please upload a video file (MP4/MOV)" : "请上传视频文件（MP4/MOV）");
        return;
      }

      if (file.size > MAX_SIZE) {
        setError(lang === "en" ? "File size must be under 100MB" : "文件大小必须小于100MB");
        return;
      }

      setFileName(file.name);
      setUploading(true);
      setUploadProgress(0);

      let finalFile = file;
      if (file.size > 15 * 1024 * 1024) {
        setUploadStage(lang === "en" ? "Compressing video..." : "视频压缩中...");
        try {
          finalFile = await compressVideoForUpload(file, (pct, stage) => {
            setUploadProgress(Math.round(pct * 0.7));
            setUploadStage(lang === "en" ? "Compressing video..." : `${stage}...`);
          });
          if (finalFile !== file) {
            const saved = ((file.size - finalFile.size) / 1024 / 1024).toFixed(1);
            setUploadStage(lang === "en" ? `Compressed (-${saved}MB)` : `已压缩 (-${saved}MB)`);
          }
        } catch {
          finalFile = file;
        }
      }

      setUploadStage(lang === "en" ? "Preparing..." : "准备中...");
      setUploadProgress(85);

      setTimeout(() => {
        setUploadProgress(100);
        setTimeout(() => {
          setUploading(false);
          setUploadStage("");
          onUploadComplete(finalFile);
        }, 300);
      }, 400);
    },
    [lang, onUploadComplete]
  );

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  const borderColor = isPro ? "border-brand-gold" : "border-brand-purple";
  const bgColor = isPro ? "bg-brand-gold/5" : "bg-brand-purple/5";

  return (
    <div className="mx-auto max-w-xl">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-300 ${
          isDragging
            ? `${borderColor}/60 ${bgColor}`
            : uploading
            ? "border-white/10 bg-white/5"
            : `border-white/20 hover:${borderColor}/40 hover:${bgColor}`
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/mov"
          onChange={handleChange}
          className="hidden"
          data-testid="e2e-upload-video-input"
        />

        {uploading ? (
          <div className="space-y-4">
            <div className="mx-auto h-16 w-16 animate-spin rounded-full border-4 border-white/10 border-t-brand-purple" />
            <p className="text-sm text-white/60">{fileName}</p>
            <div className="mx-auto h-2 w-48 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-brand-purple transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="text-xs text-white/40">
              {uploadStage || (lang === "en" ? "Preparing..." : "准备中...")} {Math.round(uploadProgress)}%
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className={`mx-auto flex h-20 w-20 items-center justify-center rounded-2xl ${isPro ? "bg-brand-gold/10" : "bg-brand-purple/10"}`}>
              <svg
                className="h-10 w-10 text-brand-gold"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
            </div>
            <div>
              <p className="text-lg font-semibold text-white">
                {lang === "en"
                  ? "Drop your swing video here"
                  : "将挥杆视频拖放到此处"}
              </p>
              <p className="mt-1 text-sm text-white/40">
                {lang === "en"
                  ? "or click to browse • MP4/MOV • Max 100MB"
                  : "或点击浏览 • MP4/MOV • 最大100MB"}
              </p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 rounded-xl bg-red-500/10 border border-red-500/20 p-3 text-center text-sm text-red-400">
          {error}
        </div>
      )}

      <div className="mt-4 text-center text-xs text-white/30">
        {lang === "en"
          ? "For best results: Record from waist height, 3m away, with good lighting"
          : "最佳效果：从腰部高度录制，距离3米，光线充足"}
      </div>
    </div>
  );
}
