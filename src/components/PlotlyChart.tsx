import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { Config, Data, Layout } from 'plotly.js';

interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
}

export function PlotlyChart({ data, layout, config, className = '' }: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const mergedLayout: Partial<Layout> = {
      autosize: true,
      paper_bgcolor: 'rgba(220, 230, 252, 0)',
      plot_bgcolor: 'rgba(234, 240, 255, 0.56)',
      colorway: ['#4D6FAF', '#72B9D8', '#7799D8', '#91B9E8', '#83BBA5'],
      font: {
        family: '"Plus Jakarta Sans", "Public Sans", system-ui, -apple-system, sans-serif',
        color: '#203653',
        size: 12,
      },
      xaxis: {
        gridcolor: 'rgba(82, 103, 131, 0.14)',
        zerolinecolor: 'rgba(77, 111, 175, 0.25)',
        linecolor: 'rgba(210, 223, 255, 0.85)',
        tickfont: { color: '#526783' },
        ...(layout?.xaxis ?? {}),
      },
      yaxis: {
        gridcolor: 'rgba(82, 103, 131, 0.14)',
        zerolinecolor: 'rgba(77, 111, 175, 0.25)',
        linecolor: 'rgba(210, 223, 255, 0.85)',
        tickfont: { color: '#526783' },
        ...(layout?.yaxis ?? {}),
      },
      margin: { l: 52, r: 24, t: 36, b: 48 },
      ...layout,
    };

    const mergedConfig: Partial<Config> = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      ...config,
    };

    void Plotly.react(el, data, mergedLayout, mergedConfig);

    const observer = new ResizeObserver(() => {
      if (containerRef.current) {
        void Plotly.Plots.resize(containerRef.current);
      }
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
      Plotly.purge(el);
    };
  }, [data, layout, config]);

  return <div ref={containerRef} className={`w-full ${className}`} />;
}
