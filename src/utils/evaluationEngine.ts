import type {
  EvaluationSummary,
  PerClassMetric,
  RocCurveSeries,
  RocPoint,
  TrainingHistoryEpoch,
  TrainingHistorySummary,
} from '@/src/types/mindtrace';

function computeBinaryRoc(yBinary: number[], scores: number[]): { points: RocPoint[]; auc: number } {
  const totalPos = yBinary.reduce((acc, v) => acc + (v === 1 ? 1 : 0), 0);
  const totalNeg = yBinary.length - totalPos;

  if (totalPos === 0 || totalNeg === 0) {
    return { points: [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }], auc: 0.5 };
  }

  const pairs = yBinary.map((label, i) => ({ label, score: scores[i] })).sort((a, b) => b.score - a.score);

  const points: RocPoint[] = [{ fpr: 0, tpr: 0 }];
  let tp = 0;
  let fp = 0;
  let i = 0;

  while (i < pairs.length) {
    const threshold = pairs[i].score;
    while (i < pairs.length && pairs[i].score === threshold) {
      if (pairs[i].label === 1) {
        tp += 1;
      } else {
        fp += 1;
      }
      i += 1;
    }
    points.push({ fpr: fp / totalNeg, tpr: tp / totalPos });
  }

  if (points[points.length - 1].fpr !== 1 || points[points.length - 1].tpr !== 1) {
    points.push({ fpr: 1, tpr: 1 });
  }

  let auc = 0;
  for (let k = 1; k < points.length; k += 1) {
    const dx = points[k].fpr - points[k - 1].fpr;
    const avgY = (points[k].tpr + points[k - 1].tpr) / 2;
    auc += dx * avgY;
  }

  return { points, auc };
}

/**
 * Deterministic Scikit-learn-equivalent evaluation metrics computed strictly
 * from actual user-uploaded held-out prediction rows.
 */
export function computeEvaluationMetrics(
  rows: Record<string, string>[],
  trueCol: string,
  predCol: string,
  fileName: string
): EvaluationSummary {
  const validRows = rows.filter((r) => r[trueCol]?.trim() && r[predCol]?.trim());
  if (validRows.length === 0) {
    throw new Error(`No valid rows found with non-empty '${trueCol}' and '${predCol}' values.`);
  }

  const yTrue = validRows.map((r) => r[trueCol].trim());
  const yPred = validRows.map((r) => r[predCol].trim());
  const labels = Array.from(new Set([...yTrue, ...yPred])).sort((a, b) => a.localeCompare(b));
  const labelIndex = new Map<string, number>(labels.map((lbl, idx) => [lbl, idx]));

  const k = labels.length;
  const cm: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
  let correct = 0;

  for (let i = 0; i < validRows.length; i += 1) {
    const tIdx = labelIndex.get(yTrue[i])!;
    const pIdx = labelIndex.get(yPred[i])!;
    cm[tIdx][pIdx] += 1;
    if (tIdx === pIdx) {
      correct += 1;
    }
  }

  const perClassMetrics: PerClassMetric[] = labels.map((lbl, idx) => {
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

  const sampleCount = validRows.length;
  const accuracy = correct / sampleCount;
  const macroPrecision = perClassMetrics.reduce((acc, m) => acc + m.precision, 0) / k;
  const macroRecall = perClassMetrics.reduce((acc, m) => acc + m.recall, 0) / k;
  const macroF1 = perClassMetrics.reduce((acc, m) => acc + m.f1Score, 0) / k;
  const weightedF1 = perClassMetrics.reduce((acc, m) => acc + m.f1Score * m.support, 0) / sampleCount;

  const sampleKeys = Object.keys(validRows[0]);
  const hasAllProbCols =
    k >= 2 &&
    labels.every((lbl) => sampleKeys.includes(`prob_${lbl}`)) &&
    validRows.every((r) => labels.every((lbl) => !Number.isNaN(Number(r[`prob_${lbl}`]))));

  const rocCurves: RocCurveSeries[] = [];
  let rocAucMacro: number | null = null;

  if (hasAllProbCols) {
    const targetLabels = k === 2 ? [labels[1]] : labels;
    for (const lbl of targetLabels) {
      const yBin = yTrue.map((y) => (y === lbl ? 1 : 0));
      const hasBothClasses = yBin.includes(1) && yBin.includes(0);
      if (!hasBothClasses) continue;
      const scores = validRows.map((r) => Number(r[`prob_${lbl}`]));
      const { points, auc } = computeBinaryRoc(yBin, scores);
      rocCurves.push({ label: lbl, auc, points });
    }
    if (rocCurves.length > 0) {
      rocAucMacro = rocCurves.reduce((acc, c) => acc + c.auc, 0) / rocCurves.length;
    }
  }

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

function parseOptionalNum(val: string | undefined): number | null {
  if (!val || val.trim() === '') return null;
  const num = Number(val);
  return Number.isFinite(num) ? num : null;
}

export function parseTrainingHistoryCsv(
  rows: Record<string, string>[],
  fileName: string
): TrainingHistorySummary {
  if (rows.length === 0) {
    throw new Error('Training history CSV contains no rows.');
  }
  const keys = Object.keys(rows[0]).map((k) => k.toLowerCase());
  const findKey = (candidates: string[]) =>
    Object.keys(rows[0]).find((k) => candidates.includes(k.toLowerCase()));

  const epochKey = findKey(['epoch', 'step']);
  if (!epochKey || !keys.some((k) => k.includes('loss'))) {
    throw new Error(
      "Training history CSV must include an 'epoch' column and at least one loss column (e.g., 'train_loss', 'eval_loss')."
    );
  }

  const trainLossKey = findKey(['train_loss', 'loss', 'training_loss']);
  const evalLossKey = findKey(['eval_loss', 'val_loss', 'validation_loss']);
  const evalAccKey = findKey(['eval_accuracy', 'val_accuracy', 'accuracy']);
  const evalF1Key = findKey(['eval_f1', 'eval_macro_f1', 'val_f1', 'f1']);

  const epochs: TrainingHistoryEpoch[] = rows
    .map((r) => ({
      epoch: Number(r[epochKey]),
      trainLoss: trainLossKey ? parseOptionalNum(r[trainLossKey]) : null,
      evalLoss: evalLossKey ? parseOptionalNum(r[evalLossKey]) : null,
      evalAccuracy: evalAccKey ? parseOptionalNum(r[evalAccKey]) : null,
      evalF1: evalF1Key ? parseOptionalNum(r[evalF1Key]) : null,
    }))
    .filter((item) => Number.isFinite(item.epoch))
    .sort((a, b) => a.epoch - b.epoch);

  if (epochs.length === 0) {
    throw new Error('No valid numeric epoch rows found in training history CSV.');
  }

  return { fileName, epochs };
}
