"use client";

import React, { Component, type ErrorInfo, type ReactNode } from "react";

const DETAILS_MAX = 4000;
const STACK_MAX = 8000;

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return `${s.slice(0, max - 20)}\n… [truncated]`;
}

type Props = {
  label: string;
  details?: Record<string, unknown>;
  children: ReactNode;
};

type State = {
  error: Error | null;
  componentStack: string | null;
};

/**
 * 渲染期错误边界：仅在子树抛错时显示完整 name/message/componentStack + 调用方摘要。
 * 不依赖后端；正常渲染零影响。
 */
export default class FrontErrorBoundary extends Component<Props, State> {
  state: State = { error: null, componentStack: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ componentStack: info.componentStack ?? null });
    console.error("[FrontErrorBoundary]", this.props.label, error, info);
  }

  private reset = (): void => {
    this.setState({ error: null, componentStack: null });
    window.location.reload();
  };

  render(): ReactNode {
    const { label, details, children } = this.props;
    const { error, componentStack } = this.state;

    if (!error) return children;

    let detailsStr = "";
    try {
      detailsStr = truncate(JSON.stringify(details ?? {}, null, 2), DETAILS_MAX);
    } catch {
      detailsStr = "[details stringify failed]";
    }

    const stack = componentStack ? truncate(componentStack.trim(), STACK_MAX) : "—";

    return (
      <div
        role="alert"
        className="rounded-xl border border-red-500/45 bg-black/75 p-3 text-left shadow-lg backdrop-blur-sm"
      >
        <p className="text-[11px] font-semibold uppercase tracking-wide text-red-300/95">
          Prov3 result renderer crashed
        </p>
        <p className="mt-1 text-[10px] text-white/50">Area: {label}</p>
        <div className="mt-2 space-y-1.5 text-[11px] leading-snug text-red-100/90">
          <p className="break-words">
            <span className="text-white/45">Name: </span>
            {error.name || "Error"}
          </p>
          <p className="whitespace-pre-wrap break-words">
            <span className="text-white/45">Message: </span>
            {error.message || "—"}
          </p>
        </div>
        {detailsStr ? (
          <div className="mt-2">
            <p className="text-[9px] font-medium uppercase tracking-wide text-orange-200/60">Details</p>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded border border-orange-500/25 bg-black/50 p-2 font-mono text-[10px] text-orange-100/85">
              {detailsStr}
            </pre>
          </div>
        ) : null}
        <div className="mt-2">
          <p className="text-[9px] font-medium uppercase tracking-wide text-white/35">Component stack</p>
          <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded border border-white/10 bg-black/40 p-2 font-mono text-[9px] text-white/55">
            {stack}
          </pre>
        </div>
        <button
          type="button"
          onClick={this.reset}
          className="mt-3 w-full rounded-lg border border-red-400/40 bg-red-500/15 py-2 text-center text-[11px] font-medium text-red-100 active:scale-[0.99]"
        >
          Retry (reload page)
        </button>
      </div>
    );
  }
}
