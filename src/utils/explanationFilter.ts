import type { PhraseAttribution, TokenAttribution } from '@/src/types/mindtrace';
export type { PhraseAttribution };

/**
 * Negation words, negative contractions, and intensity modifiers that directly alter
 * emotional polarity and must NEVER be discarded by stopword filtering.
 */
export const PROTECTED_NEGATION_AND_MODIFIER_WORDS = new Set([
  'not',
  'no',
  'never',
  'nobody',
  'nothing',
  'neither',
  'nor',
  'nowhere',
  'none',
  'hardly',
  'barely',
  'scarcely',
  'without',
  'lack',
  'lacking',
  'lacks',
  'unable',
  'cannot',
  "can't",
  "don't",
  "doesn't",
  "didn't",
  "won't",
  "wouldn't",
  "couldn't",
  "shouldn't",
  "isn't",
  "aren't",
  "wasn't",
  "weren't",
  "haven't",
  "hasn't",
  "hadn't",
  "mustn't",
  "needn't",
  "shan't",
  "mightn't",
  "ain't",
  "nothing's",
  "nobody's",
  'anymore',
  'very',
  'too',
  'completely',
  'totally',
  'always',
  'everything',
  'everyone',
  'alone',
  'only',
]);

/**
 * Common grammatical function words that do not carry standalone emotional meaning.
 * Kept intact in text passed to Transformer models, and filtered only at the
 * explanation / highlighting stage unless part of an active multi-word phrase.
 */
export const GRAMMATICAL_STOPWORDS = new Set([
  'a',
  'an',
  'the',
  'and',
  'or',
  'but',
  'if',
  'then',
  'else',
  'when',
  'while',
  'where',
  'because',
  'so',
  'than',
  'as',
  'of',
  'in',
  'on',
  'at',
  'to',
  'from',
  'by',
  'for',
  'with',
  'about',
  'into',
  'through',
  'during',
  'before',
  'after',
  'above',
  'below',
  'between',
  'under',
  'again',
  'further',
  'once',
  'is',
  'am',
  'are',
  'was',
  'were',
  'be',
  'been',
  'being',
  'have',
  'has',
  'had',
  'having',
  'do',
  'does',
  'did',
  'doing',
  'will',
  'would',
  'shall',
  'should',
  'can',
  'could',
  'may',
  'might',
  'must',
  'i',
  'me',
  'my',
  'myself',
  'we',
  'our',
  'ours',
  'ourselves',
  'you',
  'your',
  'yours',
  'yourself',
  'yourselves',
  'he',
  'him',
  'his',
  'himself',
  'she',
  'her',
  'hers',
  'herself',
  'it',
  'its',
  'itself',
  'they',
  'them',
  'their',
  'theirs',
  'themselves',
  'this',
  'that',
  'these',
  'those',
  'although',
  'though',
  'since',
  'until',
  'unless',
  'whether',
  'whereas',
  'despite',
  'yet',
  'either',
  'every',
  'ever',
  'why',
  'how',
  'what',
  'which',
  'who',
  'whom',
  'whose',
  'there',
  'here',
  'all',
  'both',
  'each',
  'few',
  'more',
  'most',
  'other',
  'some',
  'such',
  'own',
  'same',
  'just',
  'also',
  'even',
  'still',
  'already',
  'really',
  'actually',
  'basically',
  'literally',
  'get',
  'gets',
  'got',
  'getting',
  'feel',
  'feels',
  'feeling',
  'felt',
  'seem',
  'seems',
  'seemed',
  'keep',
  'keeps',
  'kept',
]);

/**
 * Connectors allowed to bridge two salient words inside a single readable phrase
 * (e.g. "unable to sleep", "nothing seems to be getting better", "out of control").
 * Clause-separating conjunctions ("and", "because", "but", "or") are intentionally excluded.
 */
const INTRA_PHRASE_BRIDGES = new Set([
  'to',
  'be',
  'am',
  'is',
  'are',
  'was',
  'were',
  'been',
  'being',
  'of',
  'in',
  'on',
  'at',
  'for',
  'with',
  'a',
  'an',
  'the',
  'my',
  'me',
  'myself',
  'so',
  'get',
  'gets',
  'getting',
  'got',
  'feel',
  'feels',
  'feeling',
  'felt',
  'seem',
  'seems',
  'seemed',
  'keep',
  'keeps',
  'kept',
]);

const SPECIAL_TOKENS = new Set([
  '[CLS]',
  '[SEP]',
  '[PAD]',
  '[UNK]',
  '[MASK]',
  '[URL]',
  '[USER]',
  '<s>',
  '</s>',
  '<pad>',
  '<unk>',
]);

export function normalizeWordForFilter(token: string): string {
  return token
    .replace(/^##/, '')
    .replace(/’/g, "'")
    .trim()
    .toLowerCase();
}

export function isProtectedNegationOrCue(token: string): boolean {
  const norm = normalizeWordForFilter(token);
  if (!norm) return false;
  if (PROTECTED_NEGATION_AND_MODIFIER_WORDS.has(norm)) return true;
  // Preserve negative contractions such as can't, don't, wouldn't, isn't
  if (/^[a-z]+n't$/.test(norm)) return true;
  return false;
}

export function isStandaloneStopwordOrPunctuation(token: string): boolean {
  const cleaned = token.replace(/^##/, '').trim();
  if (!cleaned) return true;
  if (SPECIAL_TOKENS.has(cleaned.toUpperCase()) || SPECIAL_TOKENS.has(cleaned)) return true;
  // Pure punctuation / non-alphanumeric symbol
  if (!/[A-Za-z0-9]/.test(cleaned)) return true;

  const norm = normalizeWordForFilter(cleaned);
  if (isProtectedNegationOrCue(norm)) return false;

  // Handle contractions whose base is a protected negation or content word (e.g. "I'm" -> stopword, "don't" -> protected)
  if (norm === "i'm" || norm === "it's" || norm === "that's" || norm === "there's" || norm === "we're" || norm === "they're" || norm === "you're" || norm === "he's" || norm === "she's") {
    return true;
  }

  return GRAMMATICAL_STOPWORDS.has(norm);
}

/**
 * Distribution-based attribution salience threshold.
 * Uses the spread (median + MAD) and peak magnitude of non-zero model weights
 * instead of a hardcoded arbitrary tiny epsilon like 1e-4.
 */
export function computeAdaptiveAttributionThreshold(weights: number[]): number {
  const absVals = weights.map((w) => Math.abs(w)).filter((v) => v > 1e-6);
  if (absVals.length === 0) return Number.POSITIVE_INFINITY;

  const maxAbs = Math.max(...absVals);
  const floor = 0.006;
  if (maxAbs < floor) return floor;

  const sorted = [...absVals].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  const devs = sorted.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
  const mad = devs.length % 2 === 0 ? (devs[mid - 1] + devs[mid]) / 2 : devs[mid];

  const candidate = Math.max(floor, 0.16 * maxAbs, median + 0.35 * mad);
  return Math.min(candidate, Math.max(floor, 0.52 * maxAbs));
}

export function classifyAttributionDirection(
  weight: number,
  threshold: number
): 'supporting' | 'opposing' | 'neutral' {
  if (weight >= threshold) return 'supporting';
  if (weight <= -threshold) return 'opposing';
  return 'neutral';
}

/**
 * Merges accidental contraction fragments (e.g. "don", "'", "t"), strips punctuation/special tokens,
 * filters standalone grammatical stopwords, and classifies direction using the distribution threshold.
 */
export function filterAndNormalizeAttributions(
  rawAttributions: TokenAttribution[]
): TokenAttribution[] {
  if (!rawAttributions || rawAttributions.length === 0) return [];

  // First merge adjacent contraction pieces if a tokenizer split e.g. "don" + "'" + "t"
  const coalesced: TokenAttribution[] = [];
  for (const item of rawAttributions) {
    const tok = (item.token ?? '').replace(/^##/, '').trim();
    if (!tok || SPECIAL_TOKENS.has(tok) || SPECIAL_TOKENS.has(tok.toUpperCase())) continue;

    const prev = coalesced[coalesced.length - 1];
    const touchesPrev =
      prev &&
      typeof prev.end === 'number' &&
      typeof item.start === 'number' &&
      prev.end === item.start;

    if (
      prev &&
      touchesPrev &&
      (tok === "'" || tok === '’' || /^['’][A-Za-z]+$/.test(tok) || ((prev.token.endsWith("'") || prev.token.endsWith('’')) && /^[A-Za-z]+$/.test(tok)))
    ) {
      prev.token = `${prev.token}${tok}`;
      prev.weight += item.weight;
      prev.end = item.end;
      continue;
    }

    coalesced.push({
      ...item,
      token: tok,
    });
  }

  // Keep only non-punctuation, non-stopword content/negation attributions for salience ranking
  const meaningful = coalesced.filter((a) => !isStandaloneStopwordOrPunctuation(a.token));
  const threshold = computeAdaptiveAttributionThreshold(meaningful.map((a) => a.weight));

  return meaningful.map((a) => ({
    ...a,
    direction: classifyAttributionDirection(a.weight, threshold),
  }));
}

export function getSalientWordAttributions(rawAttributions: TokenAttribution[]): TokenAttribution[] {
  return filterAndNormalizeAttributions(rawAttributions).filter((a) => a.direction !== 'neutral');
}

interface TextTokenPosition {
  word: string;
  start: number;
  end: number;
  attr?: TokenAttribution;
}

/**
 * Groups related adjacent salient words and safe intra-phrase connectors into readable phrases
 * (e.g. "feel completely alone", "nothing seems to be getting better", "unable to sleep").
 * Never bridges across sentence punctuation or clause conjunctions like "and" / "because".
 */
export function groupMeaningfulPhrases(
  text: string,
  rawAttributions: TokenAttribution[]
): PhraseAttribution[] {
  if (!text.trim() || rawAttributions.length === 0) return [];

  const filtered = filterAndNormalizeAttributions(rawAttributions);
  if (filtered.length === 0) return [];

  const threshold = computeAdaptiveAttributionThreshold(filtered.map((a) => a.weight));

  // Build ordered word positions across the text so we know what bridge words lie between salient hits
  const wordRegex = /\[URL\]|\[USER\]|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*/g;
  const positions: TextTokenPosition[] = [];
  let m: RegExpExecArray | null;
  while ((m = wordRegex.exec(text)) !== null) {
    positions.push({ word: m[0], start: m.index, end: m.index + m[0].length });
  }

  // Match filtered attributions onto positions by exact span or sequential lookup
  let seqCursor = 0;
  for (const attr of filtered) {
    if (typeof attr.start === 'number' && typeof attr.end === 'number') {
      const matchPos = positions.find(
        (p) => Math.abs(p.start - attr.start!) <= 1 && Math.abs(p.end - attr.end!) <= 1
      );
      if (matchPos) {
        matchPos.attr = attr;
        continue;
      }
    }
    const clean = attr.token.toLowerCase();
    for (let i = seqCursor; i < positions.length; i += 1) {
      if (positions[i].word.toLowerCase() === clean && !positions[i].attr) {
        positions[i].attr = attr;
        seqCursor = i + 1;
        break;
      }
    }
  }

  const phrases: PhraseAttribution[] = [];
  const usedIndices = new Set<number>();

  for (let i = 0; i < positions.length; i += 1) {
    if (usedIndices.has(i)) continue;
    const startPos = positions[i];
    if (!startPos.attr || startPos.attr.direction === 'neutral') continue;

    const groupIndices: number[] = [i];
    const startDir = startPos.attr.direction;
    const startsWithNegation = isProtectedNegationOrCue(startPos.word);

    let lastSalientIdx = i;
    let bridgeRun = 0;

    for (let j = i + 1; j < positions.length; j += 1) {
      const prev = positions[j - 1];
      const curr = positions[j];
      const gap = text.slice(prev.end, curr.start);

      // Stop on clause/sentence punctuation
      if (/[.,!?;:\n—–]/.test(gap)) break;

      const normCurr = normalizeWordForFilter(curr.word);
      const currAttr = curr.attr;

      if (currAttr && currAttr.direction !== 'neutral') {
        // Same polarity, OR negation-led clause binding ("not feeling happy", "nothing seems to be getting better")
        const compatiblePolarity =
          currAttr.direction === startDir || (startsWithNegation && j - lastSalientIdx <= 5);
        if (compatiblePolarity) {
          for (let k = lastSalientIdx + 1; k <= j; k += 1) {
            groupIndices.push(k);
          }
          lastSalientIdx = j;
          bridgeRun = 0;
          continue;
        }
        break;
      }

      // Allow up to 4 safe intra-phrase connector words (e.g. "unable to sleep", "nothing seems to be getting better")
      // or neutral non-stopword verb/modifier inside a negated clause
      const isBridge =
        INTRA_PHRASE_BRIDGES.has(normCurr) ||
        (startsWithNegation && !GRAMMATICAL_STOPWORDS.has(normCurr));

      if (isBridge && bridgeRun < 4) {
        bridgeRun += 1;
        continue;
      }

      break;
    }

    // Trim any trailing bridge words (keep only up to lastSalientIdx)
    const finalIndices = groupIndices.filter((idx) => idx <= lastSalientIdx);
    finalIndices.forEach((idx) => usedIndices.add(idx));

    const first = positions[finalIndices[0]];
    const last = positions[finalIndices[finalIndices.length - 1]];
    const phraseText = text.slice(first.start, last.end).trim();
    const constituentWeights = finalIndices
      .map((idx) => positions[idx].attr?.weight ?? 0);
    const totalWeight = constituentWeights.reduce((acc, v) => acc + v, 0);
    const dir =
      Math.abs(totalWeight) >= threshold
        ? classifyAttributionDirection(totalWeight, threshold)
        : startsWithNegation && finalIndices.length > 1
        ? startDir
        : 'neutral';

    if (dir !== 'neutral' && phraseText) {
      phrases.push({
        phrase: phraseText,
        weight: totalWeight,
        direction: dir,
        start: first.start,
        end: last.end,
        tokens: finalIndices.map((idx) => positions[idx].word),
      });
    }
  }

  return phrases.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
}
