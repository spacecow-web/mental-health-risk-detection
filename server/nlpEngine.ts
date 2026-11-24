import fs from 'fs';
import path from 'path';
import {
  AutoTokenizer,
  AutoModel,
  pipeline,
  type ZeroShotClassificationPipeline,
  type PreTrainedTokenizer,
  type PreTrainedModel,
} from '@huggingface/transformers';
import {
  classifyAttributionDirection,
  computeAdaptiveAttributionThreshold,
  groupMeaningfulPhrases,
  isStandaloneStopwordOrPunctuation,
  type PhraseAttribution as SharedPhraseAttribution,
} from '../src/utils/explanationFilter.ts';

export type TransformerModelId =
  | 'zero-shot-deberta'
  | 'distilbert-encoder'
  | 'bert-base-encoder';

export type ExplanationMethod = 'SHAP' | 'LIME';

export interface PreprocessOptions {
  mask_urls?: boolean;
  anonymize_mentions?: boolean;
  strip_hashtag_symbol?: boolean;
  normalize_whitespace?: boolean;
  lowercase?: boolean;
}

export interface TokenAttribution {
  token: string;
  weight: number;
  direction: 'supporting' | 'opposing' | 'neutral';
  start: number;
  end: number;
}

export interface FineTunedCheckpoint {
  id: TransformerModelId;
  displayName: string;
  hfModelId: string;
  labels: string[];
  embeddingDim: number;
  weights: number[][];
  biases: number[];
  trainedAtIso: string;
  datasetName: string;
  trainCount: number;
  valCount: number;
  testCount: number;
  classWeights: Record<string, number>;
}

export interface DefaultCorpusPayload {
  datasetName: string;
  datasetSource: string;
  fileName: string;
  rows: Array<{ text: string; label: string; split: string; subreddit?: string }>;
}

const DEFAULT_DATASET_CSV_PATH = path.resolve(
  process.cwd(),
  process.env.MINDTRACE_DEFAULT_DATASET || 'data/default_dataset.csv'
);
const CACHE_DIR = path.resolve(process.cwd(), 'artifacts/cache');
const DEFAULT_CORPUS_CACHE_PATH = path.join(CACHE_DIR, 'default_dreaddit_200.json');
const EVAL_STORE_DIR = path.resolve(
  process.cwd(),
  process.env.MINDTRACE_ARTIFACTS_DIR || 'artifacts/evaluations'
);
const PERSISTED_STATE_PATH = path.join(EVAL_STORE_DIR, 'persisted_state.json');

const DEFAULT_ZERO_SHOT_LABELS = [
  'Neutral',
  'Stress',
  'Anxiety',
  'Depressive Distress',
];

const HYPOTHESIS_TEMPLATE = 'This social media post expresses {}.';

const URL_REGEX = /https?:\/\/\S+|www\.\S+/gi;
const MENTION_REGEX = /(?<!\w)@[A-Za-z0-9_]{1,50}/g;
const HASHTAG_REGEX = /#([A-Za-z0-9_]+)/g;
const CONTROL_CHARS_REGEX = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g;
const MULTI_SPACE_REGEX = /\s+/g;

const IMMEDIATE_SELF_HARM_REGEX =
  /\b(suicid\w*|kill myself|end my life|take my own life|self[- ]?harm|hurt myself|harm myself|cutting myself|want to die|wish i were dead|better off dead|don'?t want to live|no reason to live|can'?t go on|ending it all|overdose|immediate danger)\b/i;

const SUPPORTIVE_CUE_REGEX =
  /\b(hopeless|worthless|overwhelmed|panic attack|depress\w*|severe anxiety)\b/i;

const CONFIDENCE_DISCLAIMER =
  'Model confidence is derived from normalized softmax logits across the supported labels—it is NOT the probability that a person has a mental health condition.';

let zeroShotPipelinePromise: Promise<ZeroShotClassificationPipeline> | null = null;
const tokenizerCache = new Map<string, Promise<PreTrainedTokenizer>>();
const encoderCache = new Map<string, Promise<PreTrainedModel>>();
const fineTunedRegistry = new Map<string, FineTunedCheckpoint>();
let activeModelKey: string = 'zero-shot-deberta';
let cachedDefaultCorpus: DefaultCorpusPayload | null = null;

interface PersistedEvalRunRecord {
  id: string;
  modelName: string;
  summary: ReturnType<typeof computeClassificationEvalSummary>;
}

interface PersistedTrainingHistoryRecord {
  fileName: string;
  epochs: Array<{
    epoch: number;
    trainLoss: number | null;
    evalLoss: number | null;
    evalAccuracy: number | null;
    evalF1: number | null;
  }>;
}

interface PersistedStateFile {
  activeModelKey: string;
  checkpoints: FineTunedCheckpoint[];
  evaluationRuns: PersistedEvalRunRecord[];
  activeEvalRunId: string | null;
  trainingHistoriesByRunId: Record<string, PersistedTrainingHistoryRecord>;
}

let persistedEvaluationRuns: PersistedEvalRunRecord[] = [];
let persistedActiveEvalRunId: string | null = null;
let persistedTrainingHistories: Record<string, PersistedTrainingHistoryRecord> = {};
let stateLoaded = false;

function ensureHydratedState() {
  if (stateLoaded) return;
  stateLoaded = true;
  try {
    if (!fs.existsSync(PERSISTED_STATE_PATH)) return;
    const raw = fs.readFileSync(PERSISTED_STATE_PATH, 'utf-8');
    const parsed = JSON.parse(raw) as Partial<PersistedStateFile>;
    if (Array.isArray(parsed.checkpoints)) {
      for (const cp of parsed.checkpoints) {
        if (cp && cp.id && Array.isArray(cp.labels) && Array.isArray(cp.weights)) {
          fineTunedRegistry.set(cp.id, cp);
        }
      }
    }
    if (typeof parsed.activeModelKey === 'string') {
      if (parsed.activeModelKey === 'zero-shot-deberta' || fineTunedRegistry.has(parsed.activeModelKey)) {
        activeModelKey = parsed.activeModelKey;
      }
    }
    if (Array.isArray(parsed.evaluationRuns)) {
      persistedEvaluationRuns = parsed.evaluationRuns;
    }
    if (typeof parsed.activeEvalRunId === 'string' || parsed.activeEvalRunId === null) {
      persistedActiveEvalRunId = parsed.activeEvalRunId ?? null;
    }
    if (parsed.trainingHistoriesByRunId && typeof parsed.trainingHistoriesByRunId === 'object') {
      persistedTrainingHistories = parsed.trainingHistoriesByRunId;
    }
  } catch (err) {
    console.warn('MindTrace: Could not load persisted evaluation state:', err);
  }
}

function savePersistedState() {
  try {
    fs.mkdirSync(EVAL_STORE_DIR, { recursive: true });
    const payload: PersistedStateFile = {
      activeModelKey,
      checkpoints: Array.from(fineTunedRegistry.values()),
      evaluationRuns: persistedEvaluationRuns,
      activeEvalRunId: persistedActiveEvalRunId,
      trainingHistoriesByRunId: persistedTrainingHistories,
    };
    fs.writeFileSync(PERSISTED_STATE_PATH, JSON.stringify(payload, null, 2), 'utf-8');
  } catch (err) {
    console.warn('MindTrace: Failed to persist evaluation state:', err);
  }
}

function upsertPersistedRun(
  run: PersistedEvalRunRecord,
  history?: PersistedTrainingHistoryRecord
) {
  ensureHydratedState();
  const idx = persistedEvaluationRuns.findIndex(
    (r) => r.id === run.id || r.modelName === run.modelName
  );
  if (idx >= 0) {
    const previousId = persistedEvaluationRuns[idx].id;
    if (previousId !== run.id && persistedTrainingHistories[previousId]) {
      delete persistedTrainingHistories[previousId];
    }
    persistedEvaluationRuns[idx] = run;
  } else {
    persistedEvaluationRuns.push(run);
  }
  persistedActiveEvalRunId = run.id;
  if (history) {
    persistedTrainingHistories[run.id] = history;
  }
  savePersistedState();
}

export function deletePersistedRun(runId: string) {
  ensureHydratedState();
  persistedEvaluationRuns = persistedEvaluationRuns.filter((r) => r.id !== runId);
  delete persistedTrainingHistories[runId];
  if (persistedActiveEvalRunId === runId) {
    persistedActiveEvalRunId = persistedEvaluationRuns[0]?.id ?? null;
  }
  savePersistedState();
  return getPersistedEvaluations();
}

export function getPersistedEvaluations() {
  ensureHydratedState();
  return {
    evaluationRuns: persistedEvaluationRuns,
    activeEvalRunId: persistedActiveEvalRunId,
    trainingHistoriesByRunId: persistedTrainingHistories,
  };
}

const HF_ENCODER_MAP: Record<
  'distilbert-encoder' | 'bert-base-encoder',
  { hfId: string; displayName: string }
> = {
  'distilbert-encoder': {
    hfId: 'Xenova/all-MiniLM-L6-v2',
    displayName: 'Distilled 6-Layer Transformer (all-MiniLM-L6-v2 / DistilBERT-class)',
  },
  'bert-base-encoder': {
    hfId: 'Xenova/bge-base-en-v1.5',
    displayName: 'BERT-Base 12-Layer Encoder (bge-base-en-v1.5, 768-dim)',
  },
};

export function preprocessText(rawText: string, opts: PreprocessOptions = {}) {
  const maskUrls = opts.mask_urls ?? true;
  const anonymizeMentions = opts.anonymize_mentions ?? true;
  const stripHashtag = opts.strip_hashtag_symbol ?? true;
  const normalizeWs = opts.normalize_whitespace ?? true;
  const lowercase = opts.lowercase ?? false;

  let text = rawText.normalize('NFKC').replace(CONTROL_CHARS_REGEX, ' ');
  let urlsMasked = 0;
  let mentionsMasked = 0;

  if (maskUrls) {
    text = text.replace(URL_REGEX, () => {
      urlsMasked += 1;
      return '[URL]';
    });
  }
  if (anonymizeMentions) {
    text = text.replace(MENTION_REGEX, () => {
      mentionsMasked += 1;
      return '[USER]';
    });
  }
  if (stripHashtag) {
    text = text.replace(HASHTAG_REGEX, '$1');
  }
  if (normalizeWs) {
    text = text.replace(MULTI_SPACE_REGEX, ' ').trim();
  }
  if (lowercase) {
    text = text.toLowerCase();
  }

  return { cleanedText: text, urlsMasked, mentionsMasked };
}

export function validateInputText(text: string): { isValid: boolean; error: string | null } {
  const trimmed = (text ?? '').trim();
  if (!trimmed) {
    return { isValid: false, error: 'Enter or paste social media text before running analysis.' };
  }
  if (trimmed.length < 10) {
    return {
      isValid: false,
      error: `Input is too short (${trimmed.length} characters). Provide at least 10 characters for contextual Transformer tokenization.`,
    };
  }
  return { isValid: true, error: null };
}

export function detectImmediateSelfHarm(text: string): boolean {
  return Boolean(text && IMMEDIATE_SELF_HARM_REGEX.test(text));
}

export function detectSupportiveGuidance(text: string, predictedLabel = ''): boolean {
  if (detectImmediateSelfHarm(text)) return true;
  if (text && SUPPORTIVE_CUE_REGEX.test(text)) return true;
  const lowered = predictedLabel.toLowerCase();
  return ['distress', 'depress', 'anxiety', 'stress', 'suicid', 'crisis', 'self-harm', 'risk'].some(
    (k) => lowered.includes(k)
  );
}

async function getZeroShotPipeline(): Promise<ZeroShotClassificationPipeline> {
  if (!zeroShotPipelinePromise) {
    zeroShotPipelinePromise = pipeline(
      'zero-shot-classification',
      'Xenova/nli-deberta-v3-xsmall',
      { dtype: 'q8' }
    );
  }
  return zeroShotPipelinePromise;
}

async function getTokenizer(hfModelId: string): Promise<PreTrainedTokenizer> {
  if (!tokenizerCache.has(hfModelId)) {
    tokenizerCache.set(hfModelId, AutoTokenizer.from_pretrained(hfModelId));
  }
  return tokenizerCache.get(hfModelId)!;
}

async function getEncoder(hfModelId: string): Promise<PreTrainedModel> {
  if (!encoderCache.has(hfModelId)) {
    encoderCache.set(hfModelId, AutoModel.from_pretrained(hfModelId, { dtype: 'q8' }));
  }
  return encoderCache.get(hfModelId)!;
}

function softmaxVec(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

function computeNormalizedEntropy(probs: number[]): number {
  if (probs.length <= 1) return 0;
  let entropy = 0;
  for (const p of probs) {
    const clipped = Math.max(p, 1e-12);
    entropy -= clipped * Math.log(clipped);
  }
  const maxEntropy = Math.log(probs.length);
  return Math.min(Math.max(entropy / maxEntropy, 0), 1);
}

async function encodeTextsBatch(
  texts: string[],
  hfModelId: string,
  batchSize = 16
): Promise<number[][]> {
  const tokenizer = await getTokenizer(hfModelId);
  const encoder = await getEncoder(hfModelId);
  const embeddings: number[][] = [];

  for (let start = 0; start < texts.length; start += batchSize) {
    const chunk = texts.slice(start, start + batchSize);
    const encoded = await tokenizer(chunk, {
      padding: true,
      truncation: true,
      max_length: 192,
    });
    const output = await encoder(encoded);
    const hidden = output.last_hidden_state;
    const [bSize, seqLen, dim] = hidden.dims;
    const data = hidden.data as Float32Array;
    const maskData = encoded.attention_mask.data as BigInt64Array | Int32Array | Float32Array;

    for (let b = 0; b < bSize; b += 1) {
      const pooled = new Float64Array(dim);
      let tokenCount = 0;
      for (let s = 0; s < seqLen; s += 1) {
        const maskVal = Number(maskData[b * seqLen + s]);
        if (maskVal > 0) {
          tokenCount += 1;
          const offset = (b * seqLen + s) * dim;
          for (let d = 0; d < dim; d += 1) {
            pooled[d] += data[offset + d];
          }
        }
      }
      const denom = Math.max(tokenCount, 1);
      let normSq = 0;
      const vec: number[] = new Array(dim);
      for (let d = 0; d < dim; d += 1) {
        const v = pooled[d] / denom;
        vec[d] = v;
        normSq += v * v;
      }
      const norm = Math.sqrt(normSq) || 1;
      for (let d = 0; d < dim; d += 1) {
        vec[d] /= norm;
      }
      embeddings.push(vec);
    }
  }

  return embeddings;
}

async function predictBatchWithCheckpoint(
  texts: string[],
  checkpoint: FineTunedCheckpoint
): Promise<{ probsList: number[][]; logitsList: number[][] }> {
  const embeddings = await encodeTextsBatch(texts, checkpoint.hfModelId, 16);
  const probsList: number[][] = [];
  const logitsList: number[][] = [];

  for (const emb of embeddings) {
    const logits = checkpoint.weights.map((wRow, c) => {
      let sum = checkpoint.biases[c];
      for (let d = 0; d < emb.length; d += 1) {
        sum += wRow[d] * emb[d];
      }
      return sum;
    });
    logitsList.push(logits);
    probsList.push(softmaxVec(logits));
  }
  return { probsList, logitsList };
}

async function predictBatchZeroShot(
  texts: string[],
  labels = DEFAULT_ZERO_SHOT_LABELS
): Promise<{ probsList: number[][]; logitsList: number[][] }> {
  const zsc = await getZeroShotPipeline();
  const probsList: number[][] = [];
  const logitsList: number[][] = [];

  for (const text of texts) {
    const res = await zsc(text, labels, {
      hypothesis_template: HYPOTHESIS_TEMPLATE,
      multi_label: false,
    });
    const out = Array.isArray(res) ? res[0] : res;
    const scoreMap = new Map<string, number>();
    out.labels.forEach((lbl: string, idx: number) => {
      scoreMap.set(lbl, out.scores[idx]);
    });
    const probs = labels.map((lbl) => scoreMap.get(lbl) ?? 0);
    const rawLogs = probs.map((p) => Math.log(Math.max(p, 1e-8)));
    const meanLog = rawLogs.reduce((a, b) => a + b, 0) / rawLogs.length;
    const logits = rawLogs.map((l) => l - meanLog);
    probsList.push(probs);
    logitsList.push(logits);
  }

  return { probsList, logitsList };
}

async function inspectTokens(cleanedText: string, hfModelId: string): Promise<string[]> {
  const tok = await getTokenizer(hfModelId);
  const subwords = tok.tokenize(cleanedText).slice(0, 254);
  return ['[CLS]', ...subwords, '[SEP]'];
}

interface WordSpan {
  word: string;
  start: number;
  end: number;
}

function extractWordSpans(text: string): WordSpan[] {
  const spans: WordSpan[] = [];
  // Keep contractions like don't / can't / won't / I'm as single whole words; skip standalone punctuation
  const regex = /\[URL\]|\[USER\]|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    spans.push({
      word: match[0],
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return spans;
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);

  for (let col = 0; col < n; col += 1) {
    let pivotRow = col;
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivotRow][col])) {
        pivotRow = r;
      }
    }
    [a[col], a[pivotRow]] = [a[pivotRow], a[col]];
    const pivot = Math.abs(a[col][col]) < 1e-10 ? 1e-10 : a[col][col];
    for (let j = col; j <= n; j += 1) {
      a[col][j] /= pivot;
    }
    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const factor = a[r][col];
      for (let j = col; j <= n; j += 1) {
        a[r][j] -= factor * a[col][j];
      }
    }
  }
  return a.map((row) => row[n]);
}

function seededRand(seedState: { s: number }): number {
  seedState.s = (seedState.s * 1664525 + 1013904223) % 4294967296;
  return seedState.s / 4294967296;
}

export async function computeModelAttributions(
  cleanedText: string,
  targetLabel: string,
  labels: string[],
  baseProbs: number[],
  method: ExplanationMethod,
  predictProbaFn: (texts: string[]) => Promise<number[][]>,
  maxAnalyzedSpans = 20
): Promise<{
  attributions: TokenAttribution[];
  phraseAttributions: SharedPhraseAttribution[];
}> {
  const allSpans = extractWordSpans(cleanedText);
  if (allSpans.length === 0) {
    return { attributions: [], phraseAttributions: [] };
  }

  const targetIdx = Math.max(0, labels.indexOf(targetLabel));
  const baseTargetProb = baseProbs[targetIdx] ?? 0;

  // Perturb meaningful content words, negations, and contractions while keeping
  // grammatical function words intact as syntactic scaffolding (preventing noisy
  // ungrammatical deltas on isolated "and", "the", "to", "is", "a").
  const contentIndices = allSpans
    .map((sp, idx) => ({ sp, idx }))
    .filter(({ sp }) => !isStandaloneStopwordOrPunctuation(sp.word))
    .map(({ idx }) => idx);

  const selectedIndices = (
    contentIndices.length > 0 ? contentIndices : allSpans.map((_, idx) => idx)
  ).slice(0, maxAnalyzedSpans);
  const k = selectedIndices.length;

  // Clean omission preserves natural sentence grammar without injecting artificial [MASK] tokens
  // that zero-shot NLI and sentence encoders were not trained to interpret.
  const buildOmittedVariant = (keepFlags: boolean[]): string => {
    const omittedAllIndices = new Set<number>();
    selectedIndices.forEach((spanIdx, candidatePos) => {
      if (!keepFlags[candidatePos]) {
        omittedAllIndices.add(spanIdx);
      }
    });

    let out = '';
    let cursor = 0;
    for (let i = 0; i < allSpans.length; i += 1) {
      const sp = allSpans[i];
      if (sp.start > cursor) {
        out += cleanedText.slice(cursor, sp.start);
      }
      if (!omittedAllIndices.has(i)) {
        out += sp.word;
      }
      cursor = sp.end;
    }
    if (cursor < cleanedText.length) {
      out += cleanedText.slice(cursor);
    }

    const collapsed = out
      .replace(/\s+/g, ' ')
      .replace(/\s+([.,!?;:])/g, '$1')
      .trim();
    return collapsed || 'neutral';
  };

  const rawWeights = new Array<number>(k).fill(0);

  if (method === 'SHAP') {
    // Antithetic Permutation Shapley estimator: averages exact marginal contributions
    // f(S U {i}) - f(S) across forward, reverse, and interleaved orderings.
    const fwd = Array.from({ length: k }, (_, i) => i);
    const rev = [...fwd].reverse();
    const permutations: number[][] = [fwd, rev];

    if (k >= 4 && k <= 10) {
      const oddEven = [...fwd.filter((i) => i % 2 === 0), ...fwd.filter((i) => i % 2 === 1)];
      const oddEvenRev = [...oddEven].reverse();
      permutations.push(oddEven, oddEvenRev);
    }

    const subsetVariants = new Map<string, string>();
    const fullKey = fwd.join(',');
    const subsetProbs = new Map<string, number>([[fullKey, baseTargetProb]]);

    for (const perm of permutations) {
      const active = new Set<number>();
      const emptyKey = '';
      if (!subsetProbs.has(emptyKey) && !subsetVariants.has(emptyKey)) {
        subsetVariants.set(emptyKey, buildOmittedVariant(Array(k).fill(false)));
      }
      for (const idx of perm) {
        active.add(idx);
        const key = Array.from(active).sort((a, b) => a - b).join(',');
        if (!subsetProbs.has(key) && !subsetVariants.has(key)) {
          const flags = Array.from({ length: k }, (_, pos) => active.has(pos));
          subsetVariants.set(key, buildOmittedVariant(flags));
        }
      }
    }

    const pendingEntries = Array.from(subsetVariants.entries());
    if (pendingEntries.length > 0) {
      const probRows = await predictProbaFn(pendingEntries.map(([, txt]) => txt));
      pendingEntries.forEach(([key], idx) => {
        subsetProbs.set(key, probRows[idx]?.[targetIdx] ?? baseTargetProb);
      });
    }

    for (const perm of permutations) {
      const active = new Set<number>();
      let prevScore = subsetProbs.get('') ?? baseTargetProb;
      for (const idx of perm) {
        active.add(idx);
        const key = Array.from(active).sort((a, b) => a - b).join(',');
        const nextScore = subsetProbs.get(key) ?? prevScore;
        rawWeights[idx] += (nextScore - prevScore) / permutations.length;
        prevScore = nextScore;
      }
    }
  } else {
    // LIME local weighted linear surrogate over meaningful content/negation spans
    const numSamples = Math.min(36, Math.max(16, k * 2 + 6));
    const rng = { s: 42 };
    const masks: boolean[][] = [];
    const sampleTexts: string[] = [];

    for (let s = 0; s < numSamples; s += 1) {
      const flags = Array.from({ length: k }, () => seededRand(rng) > 0.35);
      if (k > 1 && flags.every(Boolean)) {
        flags[s % k] = false;
      }
      if (flags.every((f) => !f)) {
        flags[s % k] = true;
      }
      masks.push(flags);
      sampleTexts.push(buildOmittedVariant(flags));
    }

    const sampleProbs = await predictProbaFn(sampleTexts);
    const dim = k + 1;
    const xtx = Array.from({ length: dim }, () => Array(dim).fill(0));
    const xty = Array(dim).fill(0);
    const sigma = Math.max(1, k * 0.45);

    for (let s = 0; s < numSamples; s += 1) {
      const maskedCount = masks[s].filter((f) => !f).length;
      const kernelWeight = Math.exp(-(maskedCount * maskedCount) / (sigma * sigma));
      const y = sampleProbs[s]?.[targetIdx] ?? baseTargetProb;
      const x = [1, ...masks[s].map((f) => (f ? 1 : 0))];

      for (let i = 0; i < dim; i += 1) {
        xty[i] += kernelWeight * x[i] * y;
        for (let j = 0; j < dim; j += 1) {
          xtx[i][j] += kernelWeight * x[i] * x[j];
        }
      }
    }
    for (let i = 1; i < dim; i += 1) {
      xtx[i][i] += 0.35;
    }
    xtx[0][0] += 1e-6;

    const coeffs = solveLinearSystem(xtx, xty);
    for (let i = 0; i < k; i += 1) {
      rawWeights[i] = coeffs[i + 1] ?? 0;
    }
  }

  const threshold = computeAdaptiveAttributionThreshold(rawWeights);
  const attributions: TokenAttribution[] = selectedIndices.map((spanIdx, i) => {
    const sp = allSpans[spanIdx];
    const weight = rawWeights[i];
    return {
      token: sp.word,
      weight,
      direction: classifyAttributionDirection(weight, threshold),
      start: sp.start,
      end: sp.end,
    };
  });

  const phraseAttributions = groupMeaningfulPhrases(cleanedText, attributions);

  return { attributions, phraseAttributions };
}

function buildGroundedSummary(
  method: ExplanationMethod,
  targetLabel: string,
  confidence: number,
  attributions: TokenAttribution[],
  phraseAttributions: SharedPhraseAttribution[]
): string {
  const phraseSup = phraseAttributions
    .filter((p) => p.direction === 'supporting')
    .slice(0, 3)
    .map((p) => `'${p.phrase}'`);
  const phraseOpp = phraseAttributions
    .filter((p) => p.direction === 'opposing')
    .slice(0, 3)
    .map((p) => `'${p.phrase}'`);

  const rankedWords = [...attributions].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const wordSup = rankedWords
    .filter((a) => a.direction === 'supporting')
    .slice(0, 4)
    .map((a) => `'${a.token}'`);
  const wordOpp = rankedWords
    .filter((a) => a.direction === 'opposing')
    .slice(0, 4)
    .map((a) => `'${a.token}'`);

  const sup = phraseSup.length > 0 ? phraseSup : wordSup;
  const opp = phraseOpp.length > 0 ? phraseOpp : wordOpp;

  return (
    `Using ${method}, the Transformer model predicted '${targetLabel}' with ${(
      confidence * 100
    ).toFixed(1)}% softmax confidence. The words and phrases that pushed the model's score most strongly toward '${targetLabel}' were ${
      sup.length ? sup.join(', ') : 'none above significance threshold'
    }. The words and phrases that pulled the score away from '${targetLabel}' were ${
      opp.length ? opp.join(', ') : 'none above significance threshold'
    }. Feature attribution indicates how the Transformer model responded to meaningful expressions in the text; it does not establish the actual psychological reasons behind a person's writing.`
  );
}

export function getServerModelStatus() {
  ensureHydratedState();
  const fineTuned = fineTunedRegistry.get(activeModelKey);
  const availableModels = [
    {
      id: 'zero-shot-deberta',
      name: 'Pre-Trained Zero-Shot NLI Transformer (nli-deberta-v3-xsmall)',
      ready: true,
      mode: 'zero-shot' as const,
      labels: DEFAULT_ZERO_SHOT_LABELS,
    },
    ...Array.from(fineTunedRegistry.values()).map((cp) => ({
      id: cp.id,
      name: `${cp.displayName} (Fine-Tuned on ${cp.datasetName})`,
      ready: true,
      mode: 'fine-tuned' as const,
      labels: cp.labels,
    })),
  ];

  return {
    ready: true,
    activeModelId: activeModelKey,
    mode: fineTuned ? ('fine-tuned' as const) : ('zero-shot' as const),
    model: fineTuned
      ? fineTuned.displayName
      : 'Xenova/nli-deberta-v3-xsmall (Zero-Shot NLI Transformer)',
    labels: fineTuned ? fineTuned.labels : DEFAULT_ZERO_SHOT_LABELS,
    availableModels,
    defaultDatasetPath: DEFAULT_DATASET_CSV_PATH,
    defaultModelPath: EVAL_STORE_DIR,
  };
}

export function setActiveServerModel(modelId: string) {
  ensureHydratedState();
  if (modelId === 'zero-shot-deberta' || fineTunedRegistry.has(modelId)) {
    activeModelKey = modelId;
    savePersistedState();
    return getServerModelStatus();
  }
  throw new Error(`Model '${modelId}' has not been trained yet.`);
}

export async function analyzeWithActiveTransformer(params: {
  text: string;
  method?: ExplanationMethod;
  modelId?: string;
  options?: PreprocessOptions;
}) {
  ensureHydratedState();
  const val = validateInputText(params.text);
  if (!val.isValid) {
    throw new Error(val.error || 'Invalid input text.');
  }

  const prep = preprocessText(params.text, params.options);
  const method: ExplanationMethod = params.method === 'LIME' ? 'LIME' : 'SHAP';
  const requestedKey = params.modelId || activeModelKey;
  const checkpoint = fineTunedRegistry.get(requestedKey);

  const labels = checkpoint ? checkpoint.labels : DEFAULT_ZERO_SHOT_LABELS;
  const hfIdForTokens = checkpoint ? checkpoint.hfModelId : 'Xenova/nli-deberta-v3-xsmall';
  const modelDisplayName = checkpoint
    ? `${checkpoint.displayName} (Fine-Tuned on ${checkpoint.datasetName})`
    : 'Xenova/nli-deberta-v3-xsmall (Zero-Shot NLI)';

  const predictProbaFn = async (texts: string[]): Promise<number[][]> => {
    if (checkpoint) {
      return (await predictBatchWithCheckpoint(texts, checkpoint)).probsList;
    }
    return (await predictBatchZeroShot(texts, labels)).probsList;
  };

  const [{ probsList, logitsList }, tokens] = await Promise.all([
    checkpoint
      ? predictBatchWithCheckpoint([prep.cleanedText], checkpoint)
      : predictBatchZeroShot([prep.cleanedText], labels),
    inspectTokens(prep.cleanedText, hfIdForTokens),
  ]);

  const probsVec = probsList[0];
  const logitsVec = logitsList[0];

  let bestIdx = 0;
  for (let i = 1; i < probsVec.length; i += 1) {
    if (probsVec[i] > probsVec[bestIdx]) bestIdx = i;
  }

  const predictedLabel = labels[bestIdx];
  const confidence = probsVec[bestIdx];
  const normalizedEntropy = computeNormalizedEntropy(probsVec);

  const logitsMap: Record<string, number> = {};
  const probMap: Record<string, number> = {};
  labels.forEach((lbl, idx) => {
    logitsMap[lbl] = logitsVec[idx];
    probMap[lbl] = probsVec[idx];
  });

  const { attributions, phraseAttributions } = await computeModelAttributions(
    prep.cleanedText,
    predictedLabel,
    labels,
    probsVec,
    method,
    predictProbaFn,
    checkpoint ? 32 : 16
  );

  const summary = buildGroundedSummary(
    method,
    predictedLabel,
    confidence,
    attributions,
    phraseAttributions
  );

  return {
    original_text: params.text,
    cleaned_text: prep.cleanedText,
    predicted_label: predictedLabel,
    confidence,
    logits: logitsMap,
    probabilities: probMap,
    normalized_entropy: normalizedEntropy,
    is_uncertain: normalizedEntropy >= 0.72,
    tokens,
    model_name: modelDisplayName,
    inference_mode: checkpoint ? 'fine-tuned' : 'zero-shot',
    confidence_disclaimer: CONFIDENCE_DISCLAIMER,
    method,
    attributions,
    phrase_attributions: phraseAttributions,
    natural_language_summary: summary,
    immediate_safety: detectImmediateSelfHarm(params.text),
    supportive_guidance: detectSupportiveGuidance(params.text, predictedLabel),
  };
}

function parseSimpleCsvRows(rawContent: string): Array<Record<string, string>> {
  const content = rawContent.replace(/^\uFEFF/, '');
  const records: string[][] = [];
  let field = '';
  let row: string[] = [];
  let inQuotes = false;

  for (let i = 0; i < content.length; i += 1) {
    const ch = content[i];
    const next = content[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(field);
      field = '';
    } else if (ch === '\r' && next === '\n') {
      row.push(field);
      records.push(row);
      row = [];
      field = '';
      i += 1;
    } else if (ch === '\n' || ch === '\r') {
      row.push(field);
      records.push(row);
      row = [];
      field = '';
    } else {
      field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    records.push(row);
  }
  const nonEmpty = records.filter((r) => r.some((c) => c.trim() !== ''));
  if (nonEmpty.length < 2) return [];
  const headers = nonEmpty[0].map((h, i) => h.trim() || `col_${i + 1}`);
  return nonEmpty.slice(1).map((cells) => {
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => {
      obj[h] = (cells[i] ?? '').trim();
    });
    return obj;
  });
}

export async function fetchDreadditCorpus(limit = 200) {
  const corpus = await loadDefaultCorpus(limit);
  return corpus.rows;
}

export async function loadDefaultCorpus(limit = 200): Promise<DefaultCorpusPayload> {
  if (cachedDefaultCorpus && cachedDefaultCorpus.rows.length >= limit) {
    return {
      ...cachedDefaultCorpus,
      rows: cachedDefaultCorpus.rows.slice(0, limit),
    };
  }

  // 1. Prefer user-configured local default dataset CSV if placed at data/default_dataset.csv
  if (fs.existsSync(DEFAULT_DATASET_CSV_PATH) && fs.statSync(DEFAULT_DATASET_CSV_PATH).isFile()) {
    const rawCsv = fs.readFileSync(DEFAULT_DATASET_CSV_PATH, 'utf-8');
    const records = parseSimpleCsvRows(rawCsv);
    const mapped = records
      .map((r) => ({
        text: String(r.text ?? r.post ?? '').trim(),
        label: String(r.label ?? r.class ?? '').trim(),
        split: String(r.split ?? 'train').trim() || 'train',
      }))
      .filter((r) => r.text.length >= 10 && r.label.length > 0);
    if (mapped.length >= 20) {
      cachedDefaultCorpus = {
        datasetName: 'Configured Default Research Dataset',
        datasetSource: `Local Backend Dataset (${path.basename(DEFAULT_DATASET_CSV_PATH)})`,
        fileName: path.basename(DEFAULT_DATASET_CSV_PATH),
        rows: mapped,
      };
      return {
        ...cachedDefaultCorpus,
        rows: cachedDefaultCorpus.rows.slice(0, limit),
      };
    }
  }

  // 2. Load from cached disk JSON if already downloaded previously
  if (fs.existsSync(DEFAULT_CORPUS_CACHE_PATH)) {
    try {
      const parsed = JSON.parse(
        fs.readFileSync(DEFAULT_CORPUS_CACHE_PATH, 'utf-8')
      ) as DefaultCorpusPayload;
      if (parsed && Array.isArray(parsed.rows) && parsed.rows.length >= 40) {
        cachedDefaultCorpus = parsed;
        return {
          ...parsed,
          rows: parsed.rows.slice(0, limit),
        };
      }
    } catch {
      // Ignore corrupt cache and re-fetch
    }
  }

  // 3. Fetch real public Dreaddit benchmark rows from Hugging Face Datasets Server (max 100 rows/request)
  async function fetchSplitRows(split: 'train' | 'test', count: number) {
    const collected: Array<{ row: { text: string; label: number; subreddit?: string } }> = [];
    let offset = 0;
    while (offset < count) {
      const batchLen = Math.min(100, count - offset);
      const url = `https://datasets-server.huggingface.co/rows?dataset=andreagasparini%2Fdreaddit&config=default&split=${split}&offset=${offset}&length=${batchLen}`;
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(
          `Default dataset could not be loaded (Hugging Face Datasets Server HTTP ${resp.status}). You may also place a CSV at data/default_dataset.csv.`
        );
      }
      const json = (await resp.json()) as {
        rows?: Array<{ row: { text: string; label: number; subreddit?: string } }>;
      };
      const batch = json.rows ?? [];
      collected.push(...batch);
      if (batch.length < batchLen) break;
      offset += batch.length;
    }
    return collected;
  }

  const trainCount = Math.min(140, Math.round(limit * 0.7));
  const testCount = Math.max(20, limit - trainCount);

  const [trainRowsRaw, testRowsRaw] = await Promise.all([
    fetchSplitRows('train', trainCount),
    fetchSplitRows('test', testCount),
  ]);

  const mapped = [
    ...trainRowsRaw.map((item) => ({
      text: String(item.row.text ?? '').trim(),
      label: item.row.label === 1 ? 'Stress' : 'Non-Stress',
      split: 'train',
      subreddit: String(item.row.subreddit ?? ''),
    })),
    ...testRowsRaw.map((item) => ({
      text: String(item.row.text ?? '').trim(),
      label: item.row.label === 1 ? 'Stress' : 'Non-Stress',
      split: 'test',
      subreddit: String(item.row.subreddit ?? ''),
    })),
  ].filter((r) => r.text.length >= 10);

  const payload: DefaultCorpusPayload = {
    datasetName: 'Dreaddit (Turcan & McKeown, 2019)',
    datasetSource: 'Hugging Face Hub (andreagasparini/dreaddit) — Peer-reviewed Reddit stress corpus',
    fileName: 'dreaddit_huggingface_sample.csv',
    rows: mapped,
  };

  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(DEFAULT_CORPUS_CACHE_PATH, JSON.stringify(payload), 'utf-8');
  } catch {
    // Non-fatal cache write error
  }

  cachedDefaultCorpus = payload;
  return {
    ...payload,
    rows: payload.rows.slice(0, limit),
  };
}

function deterministicShuffle<T>(arr: T[], seed = 42): T[] {
  const copy = [...arr];
  const rng = { s: seed };
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(seededRand(rng) * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function computeClassificationEvalSummary(
  fileName: string,
  yTrue: string[],
  yPred: string[],
  probsMatrix: number[][],
  labels: string[]
) {
  const k = labels.length;
  const labelIndex = new Map(labels.map((l, i) => [l, i]));
  const cm: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
  let correct = 0;

  for (let i = 0; i < yTrue.length; i += 1) {
    const ti = labelIndex.get(yTrue[i])!;
    const pi = labelIndex.get(yPred[i])!;
    cm[ti][pi] += 1;
    if (ti === pi) correct += 1;
  }

  const perClassMetrics = labels.map((lbl, idx) => {
    const tp = cm[idx][idx];
    let fp = 0;
    let fn = 0;
    for (let r = 0; r < k; r += 1) {
      if (r !== idx) {
        fp += cm[r][idx];
        fn += cm[idx][r];
      }
    }
    const support = tp + fn;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { label: lbl, precision, recall, f1Score, support };
  });

  const sampleCount = yTrue.length;
  const accuracy = correct / sampleCount;
  const macroPrecision = perClassMetrics.reduce((a, b) => a + b.precision, 0) / k;
  const macroRecall = perClassMetrics.reduce((a, b) => a + b.recall, 0) / k;
  const macroF1 = perClassMetrics.reduce((a, b) => a + b.f1Score, 0) / k;
  const weightedF1 =
    perClassMetrics.reduce((a, b) => a + b.f1Score * b.support, 0) / sampleCount;

  const targetIndices = k === 2 ? [1] : labels.map((_, idx) => idx);
  const rocCurves: Array<{ label: string; auc: number; points: Array<{ fpr: number; tpr: number }> }> =
    [];

  for (const cIdx of targetIndices) {
    const lbl = labels[cIdx];
    const yBin = yTrue.map((y) => (y === lbl ? 1 : 0));
    const totalPos = yBin.filter((v) => v === 1).length;
    const totalNeg = yBin.length - totalPos;
    if (totalPos === 0 || totalNeg === 0) continue;

    const pairs = yBin
      .map((y, i) => ({ y, score: probsMatrix[i][cIdx] }))
      .sort((a, b) => b.score - a.score);
    const points = [{ fpr: 0, tpr: 0 }];
    let tp = 0;
    let fp = 0;
    let i = 0;
    while (i < pairs.length) {
      const thresh = pairs[i].score;
      while (i < pairs.length && pairs[i].score === thresh) {
        if (pairs[i].y === 1) tp += 1;
        else fp += 1;
        i += 1;
      }
      points.push({ fpr: fp / totalNeg, tpr: tp / totalPos });
    }
    if (points[points.length - 1].fpr !== 1 || points[points.length - 1].tpr !== 1) {
      points.push({ fpr: 1, tpr: 1 });
    }
    let auc = 0;
    for (let m = 1; m < points.length; m += 1) {
      auc += (points[m].fpr - points[m - 1].fpr) * ((points[m].tpr + points[m - 1].tpr) / 2);
    }
    rocCurves.push({ label: lbl, auc, points });
  }

  const rocAucMacro =
    rocCurves.length > 0
      ? rocCurves.reduce((a, c) => a + c.auc, 0) / rocCurves.length
      : null;

  return {
    fileName,
    sampleCount,
    labels,
    accuracy,
    macroPrecision,
    macroRecall,
    macroF1,
    weightedF1,
    rocAucMacro,
    confusionMatrix: cm,
    perClassMetrics,
    rocCurves,
  };
}

function prepareLeakageFreeSplit(
  rows: Array<{ text: string; label: string }>,
  maxSamples = 240
) {
  const seen = new Set<string>();
  const cleanedRows: Array<{ cleanedText: string; label: string }> = [];
  for (const r of rows) {
    const rawText = (r.text ?? '').trim();
    const label = (r.label ?? '').trim();
    if (!rawText || !label) continue;
    const { cleanedText } = preprocessText(rawText);
    if (cleanedText.length < 10) continue;
    const key = cleanedText.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    cleanedRows.push({ cleanedText, label });
    if (cleanedRows.length >= maxSamples) break;
  }

  const labelsSorted = Array.from(new Set(cleanedRows.map((r) => r.label))).sort();
  if (labelsSorted.length < 2) {
    throw new Error('Evaluation/training requires at least 2 distinct label classes.');
  }

  const countsByLabel = new Map<string, number>();
  for (const r of cleanedRows) {
    countsByLabel.set(r.label, (countsByLabel.get(r.label) ?? 0) + 1);
  }
  const viableLabels = labelsSorted.filter((l) => (countsByLabel.get(l) ?? 0) >= 3);
  if (viableLabels.length < 2) {
    throw new Error('At least 2 classes need 3+ unique samples for stratified 70/15/15 splitting.');
  }

  const viableRows = cleanedRows.filter((r) => viableLabels.includes(r.label));
  const label2id = new Map(viableLabels.map((l, i) => [l, i]));

  const trainSet: Array<{ cleanedText: string; label: string; labelId: number }> = [];
  const valSet: Array<{ cleanedText: string; label: string; labelId: number }> = [];
  const testSet: Array<{ cleanedText: string; label: string; labelId: number }> = [];

  for (const lbl of viableLabels) {
    const group = deterministicShuffle(
      viableRows.filter((r) => r.label === lbl),
      42
    );
    const n = group.length;
    // Guarantee non-empty train, val, and test subsets whenever n >= 3
    const nTrain = Math.min(n - 2, Math.max(1, Math.round(n * 0.7)));
    const nVal = Math.min(n - nTrain - 1, Math.max(1, Math.round(n * 0.15)));
    const lid = label2id.get(lbl)!;

    group.slice(0, nTrain).forEach((item) => trainSet.push({ ...item, labelId: lid }));
    group.slice(nTrain, nTrain + nVal).forEach((item) => valSet.push({ ...item, labelId: lid }));
    group.slice(nTrain + nVal).forEach((item) => testSet.push({ ...item, labelId: lid }));
  }

  if (testSet.length === 0) {
    throw new Error('Held-out test split is empty; supply more labeled rows.');
  }

  return { viableLabels, viableRows, trainSet, valSet, testSet };
}

export async function evaluateActiveModelOnCorpus(params?: {
  rows?: Array<{ text: string; label: string }>;
  datasetName?: string;
  modelId?: string;
}) {
  ensureHydratedState();
  const corpus =
    params?.rows && params.rows.length > 0 ? null : await loadDefaultCorpus(200);
  const rawRows = params?.rows && params.rows.length > 0 ? params.rows : corpus!.rows;
  const datasetName = params?.datasetName || corpus?.datasetName || 'Default Research Corpus';

  const { viableLabels, testSet } = prepareLeakageFreeSplit(rawRows, 240);
  const targetModelKey = params?.modelId || activeModelKey;
  const checkpoint = fineTunedRegistry.get(targetModelKey);

  if (checkpoint) {
    const { probsList } = await predictBatchWithCheckpoint(
      testSet.map((t) => t.cleanedText),
      checkpoint
    );
    const preds = probsList.map((probs) => {
      let bestC = 0;
      for (let c = 1; c < probs.length; c += 1) {
        if (probs[c] > probs[bestC]) bestC = c;
      }
      return checkpoint.labels[bestC] ?? viableLabels[0];
    });
    const runId = `${checkpoint.id}-eval-${Date.now()}`;
    const summary = computeClassificationEvalSummary(
      `${checkpoint.displayName} · ${datasetName} (Held-Out Test)`,
      testSet.map((t) => t.label),
      preds,
      probsList,
      checkpoint.labels
    );
    const runRecord: PersistedEvalRunRecord = {
      id: runId,
      modelName:
        checkpoint.id === 'distilbert-encoder' ? 'DistilBERT (MiniLM-6L)' : 'BERT-base (BGE-12L)',
      summary,
    };
    upsertPersistedRun(runRecord);
    return {
      run: runRecord,
      runId: runRecord.id,
      modelName: runRecord.modelName,
      summary,
      persisted: getPersistedEvaluations(),
    };
  }

  // Evaluate built-in zero-shot NLI Transformer on binary Stress / Non-Stress held-out split
  const zeroShotEvalLabels = viableLabels;
  const { probsList } = await predictBatchZeroShot(
    testSet.map((t) => t.cleanedText),
    zeroShotEvalLabels
  );
  const preds = probsList.map((probs) => {
    let bestC = 0;
    for (let c = 1; c < probs.length; c += 1) {
      if (probs[c] > probs[bestC]) bestC = c;
    }
    return zeroShotEvalLabels[bestC];
  });
  const modelName = 'DeBERTa Zero-Shot NLI';
  const runId = `zero-shot-deberta-eval-${Date.now()}`;
  const summary = computeClassificationEvalSummary(
    `${modelName} · ${datasetName} (15% Held-Out Test)`,
    testSet.map((t) => t.label),
    preds,
    probsList,
    zeroShotEvalLabels
  );
  const runRecord: PersistedEvalRunRecord = { id: runId, modelName, summary };
  upsertPersistedRun(runRecord);
  return {
    run: runRecord,
    runId,
    modelName,
    summary,
    persisted: getPersistedEvaluations(),
  };
}

export async function trainAndEvaluateTransformer(params: {
  rows?: Array<{ text: string; label: string }>;
  backbone: 'distilbert-encoder' | 'bert-base-encoder';
  datasetName?: string;
  epochs?: number;
  maxSamples?: number;
}) {
  ensureHydratedState();
  const backboneInfo = HF_ENCODER_MAP[params.backbone];
  if (!backboneInfo) {
    throw new Error(`Unsupported backbone '${params.backbone}'.`);
  }

  const maxSamples = Math.min(Math.max(params.maxSamples ?? 240, 20), 600);
  const numEpochs = Math.min(Math.max(params.epochs ?? 12, 4), 25);

  const fallbackCorpus =
    !params.rows || params.rows.length === 0 ? await loadDefaultCorpus(200) : null;
  const inputRows = params.rows && params.rows.length > 0 ? params.rows : fallbackCorpus!.rows;

  // 1 & 2. Clean, deduplicate BEFORE splitting (zero data leakage), then stratified 70/15/15 split
  const { viableLabels, viableRows, trainSet, valSet, testSet } = prepareLeakageFreeSplit(
    inputRows,
    maxSamples
  );

  // 3. Extract contextual Transformer embeddings for train / val / test
  const trainEmb = await encodeTextsBatch(
    trainSet.map((r) => r.cleanedText),
    backboneInfo.hfId,
    16
  );
  const valEmb = await encodeTextsBatch(
    valSet.map((r) => r.cleanedText),
    backboneInfo.hfId,
    16
  );
  const testEmb = await encodeTextsBatch(
    testSet.map((r) => r.cleanedText),
    backboneInfo.hfId,
    16
  );

  const dim = trainEmb[0].length;
  const k = viableLabels.length;

  // 4. Inverse-frequency class weights computed strictly from training split
  const trainCounts = Array(k).fill(0);
  trainSet.forEach((r) => {
    trainCounts[r.labelId] += 1;
  });
  const classWeightsVec = trainCounts.map((c) => trainSet.length / (k * Math.max(c, 1)));
  const classWeightsMap: Record<string, number> = {};
  viableLabels.forEach((lbl, i) => {
    classWeightsMap[lbl] = classWeightsVec[i];
  });

  // 5. Train regularized multi-class softmax head with Adam + Early Stopping on val Macro-F1
  let W = Array.from({ length: k }, () => Array(dim).fill(0));
  let b = Array(k).fill(0);
  const mW = Array.from({ length: k }, () => Array(dim).fill(0));
  const vW = Array.from({ length: k }, () => Array(dim).fill(0));
  const mb = Array(k).fill(0);
  const vb = Array(k).fill(0);

  let bestValF1 = -1;
  let bestValLoss = Infinity;
  let bestW = W.map((row) => [...row]);
  let bestB = [...b];
  let patienceCounter = 0;

  const lr = 0.04;
  const l2Reg = 1e-4;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const historyEpochs: PersistedTrainingHistoryRecord['epochs'] = [];

  for (let epoch = 1; epoch <= numEpochs; epoch += 1) {
    const gradW = Array.from({ length: k }, () => Array(dim).fill(0));
    const gradB = Array(k).fill(0);
    let totalTrainLoss = 0;
    let totalWeightSum = 0;

    for (let idx = 0; idx < trainEmb.length; idx += 1) {
      const x = trainEmb[idx];
      const yTrue = trainSet[idx].labelId;
      const cw = classWeightsVec[yTrue];
      const logits = W.map((wRow, c) => {
        let s = b[c];
        for (let d = 0; d < dim; d += 1) s += wRow[d] * x[d];
        return s;
      });
      const probs = softmaxVec(logits);
      totalTrainLoss += -cw * Math.log(Math.max(probs[yTrue], 1e-12));
      totalWeightSum += cw;

      for (let c = 0; c < k; c += 1) {
        const diff = cw * (probs[c] - (c === yTrue ? 1 : 0));
        gradB[c] += diff;
        for (let d = 0; d < dim; d += 1) {
          gradW[c][d] += diff * x[d];
        }
      }
    }

    const invN = 1 / Math.max(totalWeightSum, 1);
    const tStep = epoch;
    for (let c = 0; c < k; c += 1) {
      const gb = gradB[c] * invN;
      mb[c] = beta1 * mb[c] + (1 - beta1) * gb;
      vb[c] = beta2 * vb[c] + (1 - beta2) * gb * gb;
      const mbHat = mb[c] / (1 - Math.pow(beta1, tStep));
      const vbHat = vb[c] / (1 - Math.pow(beta2, tStep));
      b[c] -= (lr * mbHat) / (Math.sqrt(vbHat) + eps);

      for (let d = 0; d < dim; d += 1) {
        const gw = gradW[c][d] * invN + l2Reg * W[c][d];
        mW[c][d] = beta1 * mW[c][d] + (1 - beta1) * gw;
        vW[c][d] = beta2 * vW[c][d] + (1 - beta2) * gw * gw;
        const mHat = mW[c][d] / (1 - Math.pow(beta1, tStep));
        const vHat = vW[c][d] / (1 - Math.pow(beta2, tStep));
        W[c][d] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }

    let valLoss = 0;
    const valPreds: string[] = [];
    const valProbsMat: number[][] = [];
    for (let idx = 0; idx < valEmb.length; idx += 1) {
      const x = valEmb[idx];
      const yTrue = valSet[idx].labelId;
      const logits = W.map((wRow, c) => {
        let s = b[c];
        for (let d = 0; d < dim; d += 1) s += wRow[d] * x[d];
        return s;
      });
      const probs = softmaxVec(logits);
      valLoss += -Math.log(Math.max(probs[yTrue], 1e-12));
      valProbsMat.push(probs);
      let bestC = 0;
      for (let c = 1; c < k; c += 1) if (probs[c] > probs[bestC]) bestC = c;
      valPreds.push(viableLabels[bestC]);
    }
    const valEval = computeClassificationEvalSummary(
      'validation',
      valSet.map((v) => v.label),
      valPreds,
      valProbsMat,
      viableLabels
    );
    const avgValLoss = valLoss / Math.max(valEmb.length, 1);

    historyEpochs.push({
      epoch,
      trainLoss: totalTrainLoss / Math.max(totalWeightSum, 1),
      evalLoss: avgValLoss,
      evalAccuracy: valEval.accuracy,
      evalF1: valEval.macroF1,
    });

    if (
      valEval.macroF1 > bestValF1 + 1e-4 ||
      (Math.abs(valEval.macroF1 - bestValF1) <= 1e-4 && avgValLoss < bestValLoss)
    ) {
      bestValF1 = valEval.macroF1;
      bestValLoss = avgValLoss;
      bestW = W.map((row) => [...row]);
      bestB = [...b];
      patienceCounter = 0;
    } else {
      patienceCounter += 1;
      if (patienceCounter >= 4 && epoch >= 6) {
        break;
      }
    }
  }

  W = bestW;
  b = bestB;

  // 6. Held-out 15% test evaluation
  const testPreds: string[] = [];
  const testProbsMat: number[][] = [];
  for (let idx = 0; idx < testEmb.length; idx += 1) {
    const x = testEmb[idx];
    const logits = W.map((wRow, c) => {
      let s = b[c];
      for (let d = 0; d < dim; d += 1) s += wRow[d] * x[d];
      return s;
    });
    const probs = softmaxVec(logits);
    testProbsMat.push(probs);
    let bestC = 0;
    for (let c = 1; c < k; c += 1) if (probs[c] > probs[bestC]) bestC = c;
    testPreds.push(viableLabels[bestC]);
  }

  const datasetLabel = params.datasetName?.trim() || fallbackCorpus?.datasetName || 'Default Corpus';
  const shortRunName =
    params.backbone === 'distilbert-encoder' ? 'DistilBERT (MiniLM-6L)' : 'BERT-base (BGE-12L)';

  const checkpoint: FineTunedCheckpoint = {
    id: params.backbone,
    displayName: shortRunName,
    hfModelId: backboneInfo.hfId,
    labels: viableLabels,
    embeddingDim: dim,
    weights: W,
    biases: b,
    trainedAtIso: new Date().toISOString(),
    datasetName: datasetLabel,
    trainCount: trainSet.length,
    valCount: valSet.length,
    testCount: testSet.length,
    classWeights: classWeightsMap,
  };

  fineTunedRegistry.set(params.backbone, checkpoint);
  activeModelKey = params.backbone;

  const evalSummary = computeClassificationEvalSummary(
    `${shortRunName} · ${datasetLabel} (15% Held-Out Test)`,
    testSet.map((t) => t.label),
    testPreds,
    testProbsMat,
    viableLabels
  );

  const runId = `${params.backbone}-${Date.now()}`;
  const trainingHistory: PersistedTrainingHistoryRecord = {
    fileName: `${shortRunName} (${datasetLabel})`,
    epochs: historyEpochs,
  };

  upsertPersistedRun({ id: runId, modelName: shortRunName, summary: evalSummary }, trainingHistory);

  return {
    runId,
    modelName: shortRunName,
    summary: evalSummary,
    trainingHistory,
    status: getServerModelStatus(),
    splitCounts: {
      totalUniqueValid: viableRows.length,
      train: trainSet.length,
      val: valSet.length,
      test: testSet.length,
    },
  };
}

