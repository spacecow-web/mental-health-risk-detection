/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useState } from 'react';
import { Sidebar } from '@/src/components/Sidebar';
import { TopBar } from '@/src/components/TopBar';
import { AboutProjectPage } from '@/src/pages/AboutProjectPage';
import { CheckTextPage } from '@/src/pages/CheckTextPage';
import { HomePage } from '@/src/pages/HomePage';
import { ProjectDetailsPage } from '@/src/pages/ProjectDetailsPage';
import { UnderstandResultPage } from '@/src/pages/UnderstandResultPage';
import type {
  CorpusRecord,
  DatasetSummary,
  EndpointConfig,
  InferenceExplanationBundle,
  LoadedCorpusData,
  NamedEvaluationRun,
  PageId,
  PreprocessingOptions,
  TrainingHistorySummary,
} from '@/src/types/mindtrace';
import { computeDatasetAnalytics } from '@/src/utils/datasetAnalytics';
import { DEFAULT_PREPROCESSING_OPTIONS } from '@/src/utils/nlpPipeline';
import { downloadPythonKitZip } from '@/src/utils/pythonKitRegistry';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [downloadingKit, setDownloadingKit] = useState(false);

  // Ephemeral in-memory analysis state preserved across page switches
  const [inputText, setInputText] = useState('');
  const [preprocessingOptions, setPreprocessingOptions] = useState<PreprocessingOptions>(
    DEFAULT_PREPROCESSING_OPTIONS
  );
  const [activeBundle, setActiveBundle] = useState<InferenceExplanationBundle | null>(null);

  const [endpointConfig, setEndpointConfig] = useState<EndpointConfig>({
    mode: 'built-in',
    url: '/api',
    status: 'connected',
    backbone: 'distilbert-base-uncased',
    activeServerModelId: 'zero-shot-deberta',
    availableServerModels: [
      {
        id: 'zero-shot-deberta',
        name: 'Pre-Trained Zero-Shot NLI Transformer (nli-deberta-v3-xsmall)',
        ready: true,
        mode: 'zero-shot',
        labels: ['Neutral', 'Stress', 'Anxiety', 'Depressive Distress'],
      },
    ],
    detectedLabels: ['Neutral', 'Stress', 'Anxiety', 'Depressive Distress'],
    lastError: null,
  });

  // Backend-persisted evaluation results & auto-loaded default dataset
  const [evaluationRuns, setEvaluationRuns] = useState<NamedEvaluationRun[]>([]);
  const [activeEvalRunId, setActiveEvalRunId] = useState<string | null>(null);
  const [trainingHistoriesByRunId, setTrainingHistoriesByRunId] = useState<
    Record<string, TrainingHistorySummary>
  >({});
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [loadedCorpus, setLoadedCorpus] = useState<LoadedCorpusData | null>(null);
  const [loadingDefaultDataset, setLoadingDefaultDataset] = useState<boolean>(true);
  const [defaultDatasetError, setDefaultDatasetError] = useState<string | null>(null);
  const [isCustomDatasetActive, setIsCustomDatasetActive] = useState<boolean>(false);

  const applyCorpusPayload = useCallback(
    (payload: {
      datasetName?: string;
      datasetSource?: string;
      fileName?: string;
      rows?: Array<{ text: string; label: string; split?: string }>;
    }) => {
      const rows: CorpusRecord[] = (payload.rows || []).map((r) => ({
        text: String(r.text ?? ''),
        label: String(r.label ?? ''),
        split: r.split ? String(r.split) : undefined,
      }));
      if (rows.length === 0) return;

      const rawRows = rows.map((r) => ({
        text: r.text,
        label: r.label,
        split: r.split ?? '',
      }));
      const summary = computeDatasetAnalytics(
        rawRows,
        'text',
        'label',
        payload.fileName || 'default_dataset.csv',
        rawRows.some((r) => Boolean(r.split)) ? 'split' : undefined,
        payload.datasetName,
        payload.datasetSource
      );
      setDatasetSummary(summary);
      setLoadedCorpus({
        datasetName: summary.datasetName,
        datasetSource: summary.datasetSource,
        fileName: summary.fileName,
        rows,
      });
      setIsCustomDatasetActive(false);
    },
    []
  );

  const loadDefaultBackendDataset = useCallback(async () => {
    setLoadingDefaultDataset(true);
    setDefaultDatasetError(null);
    try {
      const resp = await fetch('/api/datasets/default?limit=200');
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || 'Default research dataset unavailable.');
      }
      applyCorpusPayload(data);
    } catch (err) {
      setDefaultDatasetError(
        err instanceof Error ? err.message : 'Unable to load configured default dataset.'
      );
    } finally {
      setLoadingDefaultDataset(false);
    }
  }, [applyCorpusPayload]);

  useEffect(() => {
    let active = true;
    setLoadingDefaultDataset(true);
    fetch('/api/bootstrap')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => {
        if (!active || !data) return;

        if (data.modelStatus) {
          setEndpointConfig((prev) =>
            prev.mode === 'built-in'
              ? {
                  ...prev,
                  status: data.modelStatus.ready ? 'connected' : 'unreachable',
                  activeServerModelId: data.modelStatus.activeModelId || prev.activeServerModelId,
                  availableServerModels: Array.isArray(data.modelStatus.availableModels)
                    ? data.modelStatus.availableModels
                    : prev.availableServerModels,
                  detectedLabels: Array.isArray(data.modelStatus.labels)
                    ? data.modelStatus.labels
                    : prev.detectedLabels,
                  lastError: data.modelStatus.ready ? null : 'Configured model unavailable.',
                }
              : prev
          );
        }

        if (data.defaultDataset) {
          applyCorpusPayload(data.defaultDataset);
          setDefaultDatasetError(null);
        } else if (data.datasetError) {
          setDefaultDatasetError(String(data.datasetError));
        }

        if (data.persisted) {
          if (Array.isArray(data.persisted.evaluationRuns)) {
            setEvaluationRuns(data.persisted.evaluationRuns);
          }
          if (data.persisted.activeEvalRunId !== undefined) {
            setActiveEvalRunId(data.persisted.activeEvalRunId);
          }
          if (
            data.persisted.trainingHistoriesByRunId &&
            typeof data.persisted.trainingHistoriesByRunId === 'object'
          ) {
            setTrainingHistoriesByRunId(data.persisted.trainingHistoriesByRunId);
          }
        }
      })
      .catch(() => {
        if (!active) return;
        setEndpointConfig((prev) =>
          prev.mode === 'built-in'
            ? {
                ...prev,
                status: 'unreachable',
                lastError: 'Unable to reach MindTrace backend service.',
              }
            : prev
        );
      })
      .finally(() => {
        if (active) setLoadingDefaultDataset(false);
      });

    return () => {
      active = false;
    };
  }, [applyCorpusPayload]);

  const activeEvalRun =
    evaluationRuns.find((r) => r.id === activeEvalRunId) ?? evaluationRuns[0] ?? null;

  const activeTrainingHistory = activeEvalRun
    ? trainingHistoriesByRunId[activeEvalRun.id] ??
      Object.values(trainingHistoriesByRunId).at(-1) ??
      null
    : null;

  const hasSessionData = Boolean(inputText.trim() || activeBundle);

  function handleAddOrUpdateRun(run: NamedEvaluationRun, history?: TrainingHistorySummary) {
    setEvaluationRuns((prev) => {
      const idx = prev.findIndex((item) => item.id === run.id || item.modelName === run.modelName);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = run;
        return next;
      }
      return [...prev, run];
    });
    setActiveEvalRunId(run.id);
    if (history) {
      setTrainingHistoriesByRunId((prev) => ({ ...prev, [run.id]: history }));
    }
  }

  async function handleRemoveRun(runId: string) {
    setEvaluationRuns((prev) => {
      const remaining = prev.filter((r) => r.id !== runId);
      if (activeEvalRunId === runId) {
        setActiveEvalRunId(remaining[0]?.id ?? null);
      }
      return remaining;
    });
    setTrainingHistoriesByRunId((prev) => {
      const next = { ...prev };
      delete next[runId];
      return next;
    });
    try {
      await fetch(`/api/evaluations/${encodeURIComponent(runId)}`, { method: 'DELETE' });
    } catch {
      // Non-fatal sync error
    }
  }

  function handleClearSession() {
    setInputText('');
    setActiveBundle(null);
  }

  async function handleDownloadPythonKit() {
    setDownloadingKit(true);
    try {
      await downloadPythonKitZip();
    } finally {
      setDownloadingKit(false);
    }
  }

  return (
    <div className="min-h-screen flex text-[#203653]">
      <Sidebar
        activePage={activePage}
        onSelectPage={setActivePage}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar
          activePage={activePage}
          onOpenMobileMenu={() => setMobileMenuOpen(true)}
          hasSessionData={hasSessionData}
          onClearSession={handleClearSession}
        />

        <main className="flex-1 mx-auto w-full max-w-[1200px] px-5 py-6 lg:px-8 space-y-6">
          {activePage === 'home' && <HomePage onNavigate={setActivePage} />}

          {activePage === 'check-text' && (
            <CheckTextPage
              inputText={inputText}
              onChangeInputText={setInputText}
              options={preprocessingOptions}
              endpointConfig={endpointConfig}
              activeBundle={activeBundle}
              onSetActiveBundle={setActiveBundle}
              onNavigate={setActivePage}
            />
          )}

          {activePage === 'understand-result' && (
            <UnderstandResultPage activeBundle={activeBundle} onNavigate={setActivePage} />
          )}

          {activePage === 'about-project' && <AboutProjectPage onNavigate={setActivePage} />}

          {activePage === 'project-details' && (
            <ProjectDetailsPage
              inputText={inputText}
              onChangeInputText={setInputText}
              options={preprocessingOptions}
              onChangeOptions={setPreprocessingOptions}
              endpointConfig={endpointConfig}
              onUpdateEndpointConfig={setEndpointConfig}
              activeBundle={activeBundle}
              onSetActiveBundle={setActiveBundle}
              evaluationRuns={evaluationRuns}
              activeEvalRunId={activeEvalRunId}
              onAddOrUpdateRun={handleAddOrUpdateRun}
              onSelectRun={setActiveEvalRunId}
              onRemoveRun={handleRemoveRun}
              trainingHistory={activeTrainingHistory}
              onSetTrainingHistory={(hist) => {
                if (!hist) return;
                const key = activeEvalRun?.id || 'uploaded-history';
                setTrainingHistoriesByRunId((prev) => ({ ...prev, [key]: hist }));
              }}
              datasetSummary={datasetSummary}
              loadedCorpus={loadedCorpus}
              loadingDefaultDataset={loadingDefaultDataset}
              defaultDatasetError={defaultDatasetError}
              isCustomDatasetActive={isCustomDatasetActive}
              onRetryDefaultDataset={loadDefaultBackendDataset}
              onApplyCustomDataset={(summary, corpus) => {
                setDatasetSummary(summary);
                setLoadedCorpus(corpus);
                setIsCustomDatasetActive(true);
              }}
              onDownloadPythonKit={handleDownloadPythonKit}
              downloadingKit={downloadingKit}
              onNavigate={setActivePage}
            />
          )}
        </main>
      </div>
    </div>
  );
}
