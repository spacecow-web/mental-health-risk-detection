export type PageId =
  | 'home'
  | 'check-text'
  | 'understand-result'
  | 'about-project'
  | 'project-details';

export type TransformerBackbone = 'distilbert-base-uncased' | 'bert-base-uncased';

export type ExplanationMethod = 'SHAP' | 'LIME';

export interface PreprocessingOptions {
  maskUrls: boolean;
  anonymizeMentions: boolean;
  stripHashtagSymbol: boolean;
  normalizeWhitespace: boolean;
  lowercase: boolean;
}

export interface ValidationResult {
  isValid: boolean;
  errorMessage: string | null;
  warnings: string[];
  charCount: number;
  wordCount: number;
}

export interface PreprocessedResult {
  rawText: string;
  cleanedText: string;
  urlsMasked: number;
  mentionsMasked: number;
  charCountRaw: number;
  charCountCleaned: number;
  wordCountCleaned: number;
}

export interface TokenPreview {
  tokens: string[];
  subwordCount: number;
  truncated: boolean;
}

export interface TokenAttribution {
  token: string;
  weight: number;
  direction: 'supporting' | 'opposing' | 'neutral';
  start?: number;
  end?: number;
}

export interface InferenceExplanationBundle {
  source: 'local-api' | 'imported-json';
  modelName: string;
  originalText?: string;
  cleanedText: string;
  predictedLabel: string;
  confidence: number;
  logits?: Record<string, number>;
  probabilities: Record<string, number>;
  normalizedEntropy: number;
  isUncertain: boolean;
  tokens: string[];
  method: ExplanationMethod;
  attributions: TokenAttribution[];
  naturalLanguageSummary: string;
  supportiveGuidance: boolean;
  confidenceDisclaimer?: string;
  inferenceMode?: 'zero-shot' | 'fine-tuned';
  appliedOptions?: PreprocessingOptions;
}

export interface CorpusRecord {
  text: string;
  label: string;
  split?: string;
}

export interface LoadedCorpusData {
  datasetName: string;
  datasetSource: string;
  fileName: string;
  rows: CorpusRecord[];
}

export interface PerClassMetric {
  label: string;
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface RocCurveSeries {
  label: string;
  auc: number;
  points: RocPoint[];
}

export interface EvaluationSummary {
  fileName: string;
  sampleCount: number;
  labels: string[];
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  weightedF1: number;
  rocAucMacro: number | null;
  confusionMatrix: number[][];
  perClassMetrics: PerClassMetric[];
  rocCurves: RocCurveSeries[];
}

export interface NamedEvaluationRun {
  id: string;
  modelName: string;
  summary: EvaluationSummary;
}

export interface TrainingHistoryEpoch {
  epoch: number;
  trainLoss: number | null;
  evalLoss: number | null;
  evalAccuracy: number | null;
  evalF1: number | null;
}

export interface TrainingHistorySummary {
  fileName: string;
  epochs: TrainingHistoryEpoch[];
}

export interface ClassCount {
  label: string;
  count: number;
  sharePercent: number;
}

export interface HistogramBin {
  rangeLabel: string;
  minWords: number;
  maxWords: number;
  count: number;
}

export interface TermFrequency {
  term: string;
  frequency: number;
}

export interface SplitBreakdownItem {
  split: string;
  count: number;
  sharePercent: number;
}

export interface DatasetSummary {
  fileName: string;
  datasetName: string;
  datasetSource: string;
  totalRows: number;
  validRows: number;
  missingTextRows: number;
  missingLabelRows: number;
  duplicateTexts: number;
  shortTextCount: number;
  longTextCount: number;
  numClasses: number;
  imbalanceRatio: number;
  majorityClass: string;
  minorityClass: string;
  classCounts: ClassCount[];
  splitBreakdown: SplitBreakdownItem[] | null;
  lengthStats: {
    meanWords: number;
    medianWords: number;
    p95Words: number;
    maxWords: number;
    meanChars: number;
  };
  histogramBins: HistogramBin[];
  topTermsByClass: Record<string, TermFrequency[]>;
}

export interface AvailableServerModel {
  id: string;
  name: string;
  ready: boolean;
  mode: 'zero-shot' | 'fine-tuned';
  labels: string[];
}

export interface EndpointConfig {
  mode: 'built-in' | 'external-python';
  url: string;
  status: 'unconfigured' | 'connected' | 'unreachable';
  backbone: TransformerBackbone;
  activeServerModelId: string;
  availableServerModels: AvailableServerModel[];
  detectedLabels: string[];
  lastError: string | null;
}
