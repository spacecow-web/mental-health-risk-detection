import type { InferenceExplanationBundle, TokenAttribution } from '@/src/types/mindtrace';

export interface HighlightSegment {
  text: string;
  highlighted: boolean;
  alpha: number;
}

export function buildGentleHighlightSegments(
  text: string,
  attributions: TokenAttribution[]
): HighlightSegment[] {
  if (!text) return [];

  const withOffsets = attributions
    .filter((a) => typeof a.start === 'number' && typeof a.end === 'number')
    .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));

  const resolvedSpans: { start: number; end: number; attr: TokenAttribution }[] = [];

  if (withOffsets.length > 0) {
    for (const attr of withOffsets) {
      const s = attr.start!;
      const e = attr.end!;
      if (s >= 0 && e <= text.length && s < e) {
        resolvedSpans.push({ start: s, end: e, attr });
      }
    }
  } else {
    let cursor = 0;
    const lowered = text.toLowerCase();
    for (const attr of attributions) {
      const cleanTok = attr.token.replace(/^##/, '');
      if (!cleanTok) continue;
      let idx = text.indexOf(cleanTok, cursor);
      if (idx === -1) {
        idx = lowered.indexOf(cleanTok.toLowerCase(), cursor);
      }
      if (idx !== -1) {
        resolvedSpans.push({ start: idx, end: idx + cleanTok.length, attr });
        cursor = idx + cleanTok.length;
      }
    }
  }

  if (resolvedSpans.length === 0) {
    return [{ text, highlighted: false, alpha: 0 }];
  }

  const maxAbsWeight = Math.max(...attributions.map((a) => Math.abs(a.weight)), 1e-6);
  const segments: HighlightSegment[] = [];
  let pos = 0;

  resolvedSpans.forEach((span) => {
    if (span.start < pos) return;
    if (span.start > pos) {
      segments.push({ text: text.slice(pos, span.start), highlighted: false, alpha: 0 });
    }

    const slice = text.slice(span.start, span.end);
    const { weight } = span.attr;
    const isSup = weight > 1e-4;

    if (!isSup) {
      segments.push({ text: slice, highlighted: false, alpha: 0 });
    } else {
      const norm = Math.min(Math.abs(weight) / maxAbsWeight, 1);
      const alpha = 0.22 + 0.32 * norm;
      segments.push({ text: slice, highlighted: true, alpha });
    }

    pos = span.end;
  });

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

function cleanDisplayWord(token: string): string {
  return token.replace(/^##/, '').trim();
}

function uniqueWords(items: TokenAttribution[], limit: number): string[] {
  const seen = new Set<string>();
  const words: string[] = [];
  for (const item of items) {
    const cleaned = cleanDisplayWord(item.token);
    if (!cleaned || /^[^\w]+$/.test(cleaned)) continue;
    const key = cleaned.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    words.push(cleaned);
    if (words.length >= limit) break;
  }
  return words;
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

  const supporting = [...bundle.attributions]
    .filter((a) => a.weight > 1e-4)
    .sort((a, b) => b.weight - a.weight);
  const opposing = [...bundle.attributions]
    .filter((a) => a.weight < -1e-4)
    .sort((a, b) => a.weight - b.weight);

  const supportingWords = uniqueWords(supporting, 6);
  const opposingWords = uniqueWords(opposing, 4);

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
    const quoted = supportingWords.slice(0, 4).map((w) => `"${w}"`).join(', ');
    if (isNeutralLike) {
      whyItAppeared.push(
        `Words and phrases such as ${quoted} stood out to the model as calm or everyday expressions.`
      );
    } else {
      whyItAppeared.push(
        `Words and phrases such as ${quoted} stood out most strongly as expressions commonly linked with emotional weight or strain.`
      );
    }
  } else {
    whyItAppeared.push(
      'No single word stood out on its own—instead, the overall phrasing across the message shaped the result.'
    );
  }

  if (opposingWords.length > 0) {
    const oppQuoted = opposingWords.slice(0, 3).map((w) => `"${w}"`).join(', ');
    whyItAppeared.push(
      `At the same time, words like ${oppQuoted} gently balanced the reading in the opposite direction.`
    );
  }

  if (bundle.isUncertain) {
    whyItAppeared.push(
      'Because several different tones appeared together, the AI stayed cautious rather than leaning heavily toward one pattern.'
    );
  } else {
    whyItAppeared.push(
      'The AI looks at how words work together in context—not just one word by itself—before deciding which pattern fits best.'
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
