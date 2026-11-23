/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { EthicsDisclaimerBanner } from '@/src/components/EthicsBanner';
import { Sidebar } from '@/src/components/Sidebar';
import { TopBar } from '@/src/components/TopBar';
import { AboutMethodologyPage } from '@/src/pages/AboutMethodologyPage';
import { DatasetAnalyticsPage } from '@/src/pages/DatasetAnalyticsPage';
import { ExplainableAIPage } from '@/src/pages/ExplainableAIPage';
import { ModelPerformancePage } from '@/src/pages/ModelPerformancePage';
import { OverviewPage } from '@/src/pages/OverviewPage';
import { TextAnalysisPage } from '@/src/pages/TextAnalysisPage';
import type {
  DatasetSummary,
  EndpointConfig,
  InferenceExplanationBundle,
  NamedEvaluationRun,
  PageId,
  TrainingHistorySummary,
} from '@/src/types/mindtrace';
import { downloadPythonKitZip } from '@/src/utils/pythonKitRegistry';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [downloadingKit, setDownloadingKit] = useState(false);

  // Ephemeral in-memory research session state (zero default disk persistence)
  const [inputText, setInputText] = useState('');
  const [endpointConfig, setEndpointConfig] = useState<EndpointConfig>({
    url: 'http://127.0.0.1:8000',
    status: 'unconfigured',
    backbone: 'distilbert-base-uncased',
    detectedLabels: [],
    lastError: null,
  });
  const [activeBundle, setActiveBundle] = useState<InferenceExplanationBundle | null>(null);
  const [evaluationRuns, setEvaluationRuns] = useState<NamedEvaluationRun[]>([]);
  const [activeEvalRunId, setActiveEvalRunId] = useState<string | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistorySummary | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);

  const activeEvalRun =
    evaluationRuns.find((r) => r.id === activeEvalRunId) ?? evaluationRuns[0] ?? null;

  const hasSessionData = Boolean(
    inputText.trim() ||
      activeBundle ||
      evaluationRuns.length > 0 ||
      trainingHistory ||
      datasetSummary
  );

  function handleAddOrUpdateRun(run: NamedEvaluationRun) {
    setEvaluationRuns((prev) => {
      const idx = prev.findIndex((item) => item.id === run.id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = run;
        return next;
      }
      return [...prev, run];
    });
    setActiveEvalRunId(run.id);
  }

  function handleRemoveRun(runId: string) {
    setEvaluationRuns((prev) => {
      const remaining = prev.filter((r) => r.id !== runId);
      if (activeEvalRunId === runId) {
        setActiveEvalRunId(remaining[0]?.id ?? null);
      }
      return remaining;
    });
  }

  function handleClearSession() {
    setInputText('');
    setActiveBundle(null);
    setEvaluationRuns([]);
    setActiveEvalRunId(null);
    setTrainingHistory(null);
    setDatasetSummary(null);
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
    <div className="min-h-screen flex bg-[#F8FAFC] text-[#0F172A]">
      <Sidebar
        activePage={activePage}
        onSelectPage={setActivePage}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
        endpointConfig={endpointConfig}
        evaluationRuns={evaluationRuns}
        datasetSummary={datasetSummary}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar
          activePage={activePage}
          onOpenMobileMenu={() => setMobileMenuOpen(true)}
          onDownloadPythonKit={handleDownloadPythonKit}
          downloadingKit={downloadingKit}
          hasSessionData={hasSessionData}
          onClearSession={handleClearSession}
        />

        <main className="flex-1 mx-auto w-full max-w-[1200px] px-5 py-6 lg:px-8 space-y-6">
          <EthicsDisclaimerBanner />

          {activePage === 'overview' && (
            <OverviewPage
              onNavigate={setActivePage}
              endpointConfig={endpointConfig}
              activeEvalRun={activeEvalRun}
              datasetSummary={datasetSummary}
              activeBundle={activeBundle}
            />
          )}

          {activePage === 'text-analysis' && (
            <TextAnalysisPage
              inputText={inputText}
              onChangeInputText={setInputText}
              endpointConfig={endpointConfig}
              onUpdateEndpointConfig={setEndpointConfig}
              activeBundle={activeBundle}
              onSetActiveBundle={setActiveBundle}
              onNavigate={setActivePage}
            />
          )}

          {activePage === 'explainable-ai' && (
            <ExplainableAIPage
              activeBundle={activeBundle}
              onSetActiveBundle={setActiveBundle}
              onNavigate={setActivePage}
            />
          )}

          {activePage === 'model-performance' && (
            <ModelPerformancePage
              evaluationRuns={evaluationRuns}
              activeRunId={activeEvalRunId}
              onAddOrUpdateRun={handleAddOrUpdateRun}
              onSelectRun={setActiveEvalRunId}
              onRemoveRun={handleRemoveRun}
              trainingHistory={trainingHistory}
              onSetTrainingHistory={setTrainingHistory}
            />
          )}

          {activePage === 'dataset-analytics' && (
            <DatasetAnalyticsPage
              datasetSummary={datasetSummary}
              onSetDatasetSummary={setDatasetSummary}
            />
          )}

          {activePage === 'about-methodology' && (
            <AboutMethodologyPage
              onDownloadPythonKit={handleDownloadPythonKit}
              downloadingKit={downloadingKit}
            />
          )}
        </main>
      </div>
    </div>
  );
}
