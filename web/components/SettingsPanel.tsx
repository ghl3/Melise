"use client";

import type { GenParams } from "@/lib/types";

// Quality-first defaults: low temperature keeps the RL-tuned tasks
// accurate (they were selected under greedy decoding) while a modest
// top-k stops the greedy repetition loops freeform prompts fall into.
export const DEFAULT_PARAMS: GenParams = {
  temperature: 0.6,
  top_k: 40,
  max_tokens: 256,
};

function Slider({ label, hint, value, min, max, step, display, onChange, disabled }: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
  disabled: boolean;
}) {
  return (
    <label className="block">
      <span className="flex items-baseline justify-between text-xs text-dim">
        {label}
        <span className="font-mono text-moss-soft">{display}</span>
      </span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1.5 w-full accent-moss disabled:opacity-40"
      />
      <span className="block text-[10px] leading-tight text-dim/70">{hint}</span>
    </label>
  );
}

export default function SettingsPanel({ params, onChange, disabled }: {
  params: GenParams;
  onChange: (p: GenParams) => void;
  disabled: boolean;
}) {
  const isDefault =
    params.temperature === DEFAULT_PARAMS.temperature &&
    params.top_k === DEFAULT_PARAMS.top_k &&
    params.max_tokens === DEFAULT_PARAMS.max_tokens;

  return (
    <div className="mb-2 rounded-2xl border border-line bg-panel/80 p-4">
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-3">
        <Slider
          label="temperature"
          hint="0 = always the likeliest word; higher = adventurous"
          value={params.temperature}
          min={0}
          max={1.5}
          step={0.05}
          display={params.temperature === 0 ? "greedy" : params.temperature.toFixed(2)}
          onChange={(v) => onChange({ ...params, temperature: v })}
          disabled={disabled}
        />
        <Slider
          label="top-k"
          hint="sample from the k likeliest tokens; 0 = no limit"
          value={params.top_k}
          min={0}
          max={200}
          step={5}
          display={params.top_k === 0 ? "off" : String(params.top_k)}
          onChange={(v) => onChange({ ...params, top_k: v })}
          disabled={disabled}
        />
        <Slider
          label="max tokens"
          hint="reply length cap"
          value={params.max_tokens}
          min={32}
          max={512}
          step={32}
          display={String(params.max_tokens)}
          onChange={(v) => onChange({ ...params, max_tokens: v })}
          disabled={disabled}
        />
      </div>
      {!isDefault && (
        <button
          onClick={() => onChange(DEFAULT_PARAMS)}
          disabled={disabled}
          className="mt-3 text-[11px] text-dim underline-offset-4 transition-colors hover:text-moss-soft hover:underline disabled:opacity-40"
        >
          reset to defaults
        </button>
      )}
    </div>
  );
}
