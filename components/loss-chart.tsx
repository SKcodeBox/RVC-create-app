"use client";

import { useId, useState, type KeyboardEvent, type PointerEvent } from "react";

const WIDTH = 640;
const HEIGHT = 200;
const PAD = { top: 16, right: 16, bottom: 28, left: 48 };

function niceRange(values: number[]): [number, number] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [min * 0.9, max * 1.1 || 1];
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function ticks(min: number, max: number, count: number) {
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

/**
 * エポックごとの値（誤差など）の折れ線グラフ。1つの系列だけを描く（凡例は見出しが兼ねる）。
 * マウス・キーボードで点を選ぶと、その値を表示する。highlight のエポックには印を付ける。
 */
export function LossChart({
  points,
  label,
  highlight,
  highlightLabel = "おすすめ",
  format = (value) => value.toFixed(3),
}: {
  points: [number, number][];
  /** 画面の読み上げ用の説明 */
  label: string;
  highlight?: number | null;
  highlightLabel?: string;
  format?: (value: number) => string;
}) {
  const titleId = useId();
  const [active, setActive] = useState<number | null>(null);
  if (points.length === 0) return null;

  const epochs = points.map(([epoch]) => epoch);
  const values = points.map(([, value]) => value);
  const [minY, maxY] = niceRange(values);
  const minX = Math.min(...epochs);
  const maxX = Math.max(...epochs) === minX ? minX + 1 : Math.max(...epochs);
  const x = (epoch: number) => PAD.left + ((epoch - minX) / (maxX - minX)) * (WIDTH - PAD.left - PAD.right);
  const y = (value: number) => PAD.top + (1 - (value - minY) / (maxY - minY)) * (HEIGHT - PAD.top - PAD.bottom);
  const path = points.map(([epoch, value], i) => `${i === 0 ? "M" : "L"}${x(epoch).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const showMarkers = points.length <= 60;
  const highlighted = highlight != null ? points.find(([epoch]) => epoch === highlight) : undefined;
  const current = active != null ? points[active] : null;

  function nearest(clientX: number, rect: DOMRect) {
    const svgX = ((clientX - rect.left) / rect.width) * WIDTH;
    let best = 0;
    for (let i = 1; i < points.length; i += 1) {
      if (Math.abs(x(points[i][0]) - svgX) < Math.abs(x(points[best][0]) - svgX)) best = i;
    }
    return best;
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    setActive(nearest(event.clientX, event.currentTarget.getBoundingClientRect()));
  }

  function onKeyDown(event: KeyboardEvent<SVGSVGElement>) {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      setActive((value) => Math.min(points.length - 1, Math.max(0, (value ?? (step > 0 ? -1 : points.length)) + step)));
    }
  }

  const tooltipLeft = current ? (x(current[0]) / WIDTH) * 100 : 0;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full touch-none select-none focus-visible:outline-2 focus-visible:outline-violet-500"
        role="img"
        aria-labelledby={titleId}
        tabIndex={0}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setActive(null)}
        onFocus={() => setActive((value) => value ?? points.length - 1)}
        onBlur={() => setActive(null)}
        onKeyDown={onKeyDown}
      >
        <title id={titleId}>{label}</title>
        {ticks(minY, maxY, 3).map((value) => (
          <g key={value}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(value)}
              y2={y(value)}
              className="stroke-zinc-200 dark:stroke-zinc-800"
              strokeWidth={1}
            />
            <text x={PAD.left - 8} y={y(value) + 4} textAnchor="end" className="fill-zinc-500 text-[11px] tabular-nums dark:fill-zinc-400">
              {format(value)}
            </text>
          </g>
        ))}
        {[minX, maxX].map((epoch) => (
          <text
            key={epoch}
            x={x(epoch)}
            y={HEIGHT - 8}
            textAnchor={epoch === minX ? "start" : "end"}
            className="fill-zinc-500 text-[11px] tabular-nums dark:fill-zinc-400"
          >
            {epoch}エポック
          </text>
        ))}
        {current && (
          <line
            x1={x(current[0])}
            x2={x(current[0])}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
            className="stroke-zinc-400 dark:stroke-zinc-500"
            strokeWidth={1}
          />
        )}
        <path d={path} fill="none" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" className="stroke-violet-600 dark:stroke-violet-400" />
        {showMarkers &&
          points.map(([epoch, value]) => (
            <circle
              key={epoch}
              cx={x(epoch)}
              cy={y(value)}
              r={4}
              strokeWidth={2}
              className="fill-violet-600 stroke-white dark:fill-violet-400 dark:stroke-zinc-900"
            />
          ))}
        {highlighted && (
          <g>
            <circle
              cx={x(highlighted[0])}
              cy={y(highlighted[1])}
              r={8}
              fill="none"
              strokeWidth={2}
              className="stroke-emerald-600 dark:stroke-emerald-400"
            />
            <text
              x={x(highlighted[0])}
              y={Math.max(12, y(highlighted[1]) - 12)}
              textAnchor={x(highlighted[0]) > WIDTH - 80 ? "end" : "middle"}
              className="fill-zinc-700 text-[11px] font-medium dark:fill-zinc-200"
            >
              {highlightLabel}（{highlighted[0]}）
            </text>
          </g>
        )}
        {current && (
          <circle
            cx={x(current[0])}
            cy={y(current[1])}
            r={5}
            strokeWidth={2}
            className="fill-violet-600 stroke-white dark:fill-violet-400 dark:stroke-zinc-900"
          />
        )}
      </svg>
      {current && (
        <div
          className="pointer-events-none absolute top-0 -translate-x-1/2 rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-xs shadow-sm dark:border-zinc-700 dark:bg-zinc-900"
          style={{ left: `${Math.min(88, Math.max(12, tooltipLeft))}%` }}
          aria-live="polite"
        >
          <p className="font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">{format(current[1])}</p>
          <p className="text-zinc-500 dark:text-zinc-400">{current[0]}エポック</p>
        </div>
      )}
    </div>
  );
}
