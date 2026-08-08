"use client";

import type { DoneStats, GenParams } from "@/lib/types";

function Num({ label, value, step, min, max, onChange, disabled }: {
  label: string;
  value: number;
  step: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  disabled: boolean;
}) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-dim">
      {label}
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        disabled={disabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)));
        }}
        className="w-16 rounded-md border border-line bg-panel px-1.5 py-1 font-mono text-xs text-ink outline-none focus:border-moss"
      />
    </label>
  );
}

export default function ControlsBar({ params, onChange, stats, disabled }: {
  params: GenParams;
  onChange: (p: GenParams) => void;
  stats: DoneStats | null;
  disabled: boolean;
}) {
  return (
    <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
      <div className="flex flex-wrap items-center gap-4">
        <Num label="temp" value={params.temperature} step={0.1} min={0} max={2}
          onChange={(v) => onChange({ ...params, temperature: v })} disabled={disabled} />
        <Num label="top-k" value={params.top_k} step={1} min={0} max={4096}
          onChange={(v) => onChange({ ...params, top_k: v })} disabled={disabled} />
        <Num label="max tok" value={params.max_tokens} step={32} min={16} max={512}
          onChange={(v) => onChange({ ...params, max_tokens: v })} disabled={disabled} />
      </div>
      {stats && (
        <p className="font-mono text-[11px] text-dim">
          {stats.tokens} tok · {stats.tok_per_s} tok/s ·{" "}
          {stats.stopped ? "stopped" : stats.truncated ? (
            <span className="text-alert">time-capped</span>
          ) : (
            "max length"
          )}
          {stats.dropped_turns > 0 && (
            <span className="text-alert"> · {stats.dropped_turns} old turn(s) dropped</span>
          )}
        </p>
      )}
    </div>
  );
}
