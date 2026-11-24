import type {
  InferenceExplanationBundle,
  PhraseAttribution,
  TokenAttribution,
} from '@/src/types/mindtrace';
import {
  filterAndNormalizeAttributions,
  groupMeaningfulPhrases,
} from '@/src/utils/explanationFilter';

export interface HighlightSegment {
  text: string;
  highlighted: boolean;
  alpha: number;
}

export function buildGentleHighlightSegments(
  text: string,
  attributions: TokenAttribution[],
  phraseAttributions?: PhraseAttribution[]
): HighlightSegment[] {
  if (!text) return [];

  const filteredWords = filterAndNormalizeAttributions(attributions);
  const phrases =
    phraseAttributions && phraseAttributions.length > 0
      ? phraseAttributions
      : groupMeaningfulPhrases(text, filteredWords);

  // Prefer coherent phrase spans when available, falling back to filtered salient word spans
  const activeSpans: Array<{ start: number; end: number; weight: number }> = [];

  for (const ph of phrases) {
    if (ph.direction === 'supporting' && ph.start >= 0 && ph.end <= text.length && ph.start < ph.end) {
      activeSpans.push({ start: ph.start, end: ph.end, weight: ph.weight });
    }
  }

  if (activeSpans.length === 0) {
    for (const w of filteredWords) {
      if (
        w.direction === 'supporting' &&
        typeof w.start === 'number' &&
        typeof w.end === 'number' &&
        w.start >= 0 &&
        w.end <= text.length &&
        w.start < w.end
      ) {
        activeSpans.push({ start: w.start, end: w.end, weight: w.weight });
      }
    }
  }

  activeSpans.sort((a, b) => a.start - b.start);

  // De-overlap spans cleanly
  const resolvedSpans: Array<{ start: number; end: number; weight: number }> = [];
  for (const sp of activeSpans) {
    const prev = resolvedSpans[resolvedSpans.length - 1];
    if (prev && sp.start < prev.end) continue;
    resolvedSpans.push(sp);
  }

  if (resolvedSpans.length === 0) {
    return [{ text, highlighted: false, alpha: 0 }];
  }

  const maxAbsWeight = Math.max(...resolvedSpans.map((s) => Math.abs(s.weight)), 1e-6);
  const segments: HighlightSegment[] = [];
  let pos = 0;

  for (const span of resolvedSpans) {
    if (span.start > pos) {
      segments.push({ text: text.slice(pos, span.start), highlighted: false, alpha: 0 });
    }

    const slice = text.slice(span.start, span.end);
    const norm = Math.min(Math.abs(span.weight) / maxAbsWeight, 1);
    const alpha = 0.22 + 0.32 * norm;
    segments.push({ text: slice, highlighted: true, alpha });
    pos = span.end;
  }

  if (pos < text.length) {
    segments.push({ text: text.slice(pos), highlighted: false, alpha: 0 });
  }

  return segments;
}

export interface PlainLanguageInterpretation {
  headline: string;
  noticedSummary: string;
  patternTone: 'calm' | 'elevated' | 'uncertain';
  supportingWords: string[];
  opposingWords: string[];
  whyItAppeared: string[];
  whatItMeans: string[];
}

function uniqueExpressions(values: string[], limit: number): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of values) {
    const cleaned = raw.replace(/^##/, '').trim();
    if (!cleaned || !/[A-Za-z0-9]/.test(cleaned)) continue;
    const key = cleaned.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(cleaned);
    if (out.length >= limit) break;
  }
  return out;
}

export function createPlainLanguageInterpretation(
  bundle: InferenceExplanationBundle
): PlainLanguageInterpretation {
  const labelLower = bundle.predictedLabel.toLowerCase();
  const isNeutralLike =
    labelLower.includes('neutral') ||
    labelLower.includes('non-stress') ||
    labelLower === 'none' ||
    labelLower === '0';

  const sourceText = bundle.cleanedText || bundle.originalText || '';
  const normalizedWords = filterAndNormalizeAttributions(bundle.attributions);
  const phrases =
    bundle.phraseAttributions && bundle.phraseAttributions.length > 0
      ? bundle.phraseAttributions
      : groupMeaningfulPhrases(sourceText, normalizedWords);

  const supportingPhrases = [...phrases]
    .filter((p) => p.direction === 'supporting')
    .sort((a, b) => b.weight - a.weight)
    .map((p) => p.phrase);

  const opposingPhrases = [...phrases]
    .filter((p) => p.direction === 'opposing')
    .sort((a, b) => a.weight - b.weight)
    .map((p) => p.phrase);

  const supportingWordFallback = [...normalizedWords]
    .filter((a) => a.direction === 'supporting')
    .sort((a, b) => b.weight - a.weight)
    .map((a) => a.token);

  const opposingWordFallback = [...normalizedWords]
    .filter((a) => a.direction === 'opposing')
    .sort((a, b) => a.weight - b.weight)
    .map((a) => a.token);

  const supportingWords = uniqueExpressions(
    supportingPhrases.length > 0 ? supportingPhrases : supportingWordFallback,
    6
  );
  const opposingWords = uniqueExpressions(
    opposingPhrases.length > 0 ? opposingPhrases : opposingWordFallback,
    4
  );

  let headline = 'Your text appears mostly calm and everyday in tone';
  let noticedSummary =
    'Based on the phrasing in what you shared, the AI mostly recognized ordinary, balanced language rather than strong signs of emotional strain.';
  let patternTone: PlainLanguageInterpretation['patternTone'] = 'calm';

  if (bundle.isUncertain) {
    patternTone = 'uncertain';
    headline = 'Your text contains a mix of different emotional tones';
    noticedSummary =
      'The words in your message do not point strongly in a single direction. Parts of what you wrote feel balanced, while other phrases carry a little more weight or tension.';
  } else if (!isNeutralLike) {
    patternTone = 'elevated';
    if (labelLower.includes('stress')) {
      headline = 'Your text reflects signs of pressure or feeling stretched thin';
      noticedSummary =
        'The AI noticed wording often used when someone is carrying heavy pressure, fatigue, or day-to-day overwhelm.';
    } else if (labelLower.includes('anxiety') || labelLower.includes('worry')) {
      headline = 'Your text includes phrasing often linked with worry or tension';
      noticedSummary =
        'The AI picked up on words and expressions that frequently appear when someone feels uneasy, tense, or stuck in anxious thoughts.';
    } else {
      headline = 'Your text carries signs of heavy feelings or emotional strain';
      noticedSummary =
        'The AI noticed patterns in your wording that often show up when someone is feeling worn down, discouraged, or emotionally overwhelmed.';
    }
  }

  const whyItAppeared: string[] = [];

  if (supportingWords.length > 0) {
    const quoted = supportingWords
      .slice(0, 4)
      .map((w) => `"${w}"`)
      .join(', ');
    if (isNeutralLike) {
      whyItAppeared.push(
        `Expressions such as ${quoted} stood out to the model as calm or everyday wording.`
      );
    } else {
      whyItAppeared.push(
        `Expressions such as ${quoted} stood out most strongly as wording commonly linked with emotional weight or strain.`
      );
    }
  } else {
    whyItAppeared.push(
      'No single phrase stood out strongly on its own—instead, the overall phrasing across the message shaped the result.'
    );
  }

  if (opposingWords.length > 0) {
    const oppQuoted = opposingWords
      .slice(0, 3)
      .map((w) => `"${w}"`)
      .join(', ');
    whyItAppeared.push(
      `At the same time, expressions like ${oppQuoted} gently balanced the reading in the opposite direction.`
    );
  }

  if (bundle.isUncertain) {
    whyItAppeared.push(
      'Because several different tones appeared together, the AI stayed cautious rather than leaning heavily toward one pattern.'
    );
  } else {
    whyItAppeared.push(
      'The AI looks at how words work together in context—including negations and full phrases—rather than isolated filler words.'
    );
  }

  const whatItMeans = [
    'This reflection only describes patterns in how your text is phrased—it is not a diagnosis and cannot tell what someone is experiencing clinically.',
    'Language models can misread sarcasm, song lyrics, quotes, or everyday venting, so it is completely normal if a reflection does not capture everything you feel.',
    'If what you wrote reflects something heavy you are going through right now, talking with a trusted friend, family member, or supportive professional can help.',
  ];

  return {
    headline,
    noticedSummary,
    patternTone,
    supportingWords,
    opposingWords,
    whyItAppeared,
    whatItMeans,
  };
}
