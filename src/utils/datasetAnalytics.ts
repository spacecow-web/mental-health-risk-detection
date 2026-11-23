import type {
  ClassCount,
  DatasetSummary,
  HistogramBin,
  SplitBreakdownItem,
  TermFrequency,
} from '@/src/types/mindtrace';

const STOPWORDS = new Set([
  'the', 'and', 'for', 'that', 'this', 'with', 'you', 'are', 'was', 'have', 'not', 'but',
  'from', 'they', 'his', 'her', 'she', 'him', 'just', 'all', 'like', 'what', 'when', 'out',
  'can', 'about', 'been', 'had', 'has', 'were', 'would', 'there', 'their', 'into', 'will',
  'your', 'more', 'very', 'because', 'some', 'than', 'them', 'also', 'only', 'could', 'really',
]);

function quantile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) return 0;
  const pos = (sortedValues.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sortedValues[base + 1] !== undefined) {
    return sortedValues[base] + rest * (sortedValues[base + 1] - sortedValues[base]);
  }
  return sortedValues[base];
}

export function computeDatasetAnalytics(
  rows: Record<string, string>[],
  textCol: string,
  labelCol: string,
  fileName: string,
  splitCol?: string
): DatasetSummary {
  const totalRows = rows.length;
  let missingTextRows = 0;
  let missingLabelRows = 0;

  const validPairs: { text: string; label: string; split?: string }[] = [];
  for (const row of rows) {
    const rawText = (row[textCol] ?? '').trim();
    const rawLabel = (row[labelCol] ?? '').trim();
    const rawSplit = splitCol ? (row[splitCol] ?? '').trim() : '';
    if (!rawText) missingTextRows += 1;
    if (!rawLabel) missingLabelRows += 1;
    if (rawText && rawLabel) {
      validPairs.push({
        text: rawText,
        label: rawLabel,
        split: rawSplit || undefined,
      });
    }
  }

  if (validPairs.length === 0) {
    throw new Error(`No valid rows contain both non-empty '${textCol}' and '${labelCol}'.`);
  }

  const seenTexts = new Set<string>();
  let duplicateTexts = 0;
  let shortTextCount = 0;
  let longTextCount = 0;
  const labelCounter = new Map<string, number>();
  const splitCounter = new Map<string, number>();
  const wordCounts: number[] = [];
  const charCounts: number[] = [];
  const termCountersByClass = new Map<string, Map<string, number>>();

  for (const item of validPairs) {
    if (seenTexts.has(item.text)) {
      duplicateTexts += 1;
    } else {
      seenTexts.add(item.text);
    }

    labelCounter.set(item.label, (labelCounter.get(item.label) ?? 0) + 1);
    if (item.split) {
      splitCounter.set(item.split, (splitCounter.get(item.split) ?? 0) + 1);
    }

    const words = item.text.split(/\s+/).filter(Boolean);
    if (words.length < 4) shortTextCount += 1;
    if (words.length > 200) longTextCount += 1;

    wordCounts.push(words.length);
    charCounts.push(item.text.length);

    if (!termCountersByClass.has(item.label)) {
      termCountersByClass.set(item.label, new Map());
    }
    const clsMap = termCountersByClass.get(item.label)!;
    const tokens = item.text.toLowerCase().match(/[a-z]{3,}/g) ?? [];
    for (const tok of tokens) {
      if (!STOPWORDS.has(tok)) {
        clsMap.set(tok, (clsMap.get(tok) ?? 0) + 1);
      }
    }
  }

  const classCounts: ClassCount[] = Array.from(labelCounter.entries())
    .map(([label, count]) => ({
      label,
      count,
      sharePercent: (count / validPairs.length) * 100,
    }))
    .sort((a, b) => b.count - a.count);

  const splitBreakdown: SplitBreakdownItem[] | null =
    splitCounter.size > 0
      ? Array.from(splitCounter.entries())
          .map(([split, count]) => ({
            split,
            count,
            sharePercent: (count / validPairs.length) * 100,
          }))
          .sort((a, b) => b.count - a.count)
      : null;

  const sortedWords = [...wordCounts].sort((a, b) => a - b);
  const meanWords = wordCounts.reduce((a, b) => a + b, 0) / wordCounts.length;
  const meanChars = charCounts.reduce((a, b) => a + b, 0) / charCounts.length;
  const maxWords = sortedWords[sortedWords.length - 1] ?? 0;

  const binCount = Math.min(8, Math.max(4, Math.ceil(Math.sqrt(validPairs.length))));
  const binWidth = Math.max(1, Math.ceil((maxWords + 1) / binCount));
  const histogramBins: HistogramBin[] = Array.from({ length: binCount }, (_, idx) => {
    const minW = idx * binWidth;
    const maxW = (idx + 1) * binWidth - 1;
    return {
      rangeLabel: `${minW}–${maxW}`,
      minWords: minW,
      maxWords: maxW,
      count: 0,
    };
  });

  for (const wc of wordCounts) {
    const binIdx = Math.min(binCount - 1, Math.floor(wc / binWidth));
    histogramBins[binIdx].count += 1;
  }

  const topTermsByClass: Record<string, TermFrequency[]> = {};
  for (const [label, termMap] of termCountersByClass.entries()) {
    topTermsByClass[label] = Array.from(termMap.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([term, frequency]) => ({ term, frequency }));
  }

  return {
    fileName,
    totalRows,
    validRows: validPairs.length,
    missingTextRows,
    missingLabelRows,
    duplicateTexts,
    shortTextCount,
    longTextCount,
    numClasses: classCounts.length,
    classCounts,
    splitBreakdown,
    lengthStats: {
      meanWords,
      medianWords: quantile(sortedWords, 0.5),
      p95Words: quantile(sortedWords, 0.95),
      maxWords,
      meanChars,
    },
    histogramBins,
    topTermsByClass,
  };
}
