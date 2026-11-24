import type {
  ExplanationMethod,
  PreprocessedResult,
  PreprocessingOptions,
  TokenAttribution,
  TokenPreview,
  ValidationResult,
} from '@/src/types/mindtrace';
import { filterAndNormalizeAttributions } from '@/src/utils/explanationFilter';

const URL_REGEX = /https?:\/\/\S+|www\.\S+/gi;
const MENTION_REGEX = /(?<!\w)@[A-Za-z0-9_]{1,50}/g;
const HASHTAG_REGEX = /#([A-Za-z0-9_]+)/g;
const CONTROL_CHARS_REGEX = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g;
const MULTI_SPACE_REGEX = /\s+/g;

const IMMEDIATE_SELF_HARM_REGEX =
  /\b(suicid\w*|kill myself|end my life|take my own life|self[- ]?harm|hurt myself|harm myself|cutting myself|want to die|wish i were dead|better off dead|don'?t want to live|no reason to live|can'?t go on|ending it all|overdose|immediate danger)\b/i;

const SUPPORTIVE_CUE_REGEX =
  /\b(hopeless|worthless|overwhelmed|panic attack|depress\w*|severe anxiety)\b/i;

const COMMON_SUFFIXES = ['ation', 'ments', 'ness', 'ing', 'ers', 'ies', 'ied', 'ed', 'ly', 'es', 's'];

export const DEFAULT_PREPROCESSING_OPTIONS: PreprocessingOptions = {
  maskUrls: true,
  anonymizeMentions: true,
  stripHashtagSymbol: true,
  normalizeWhitespace: true,
  lowercase: false,
};

export function validateInputText(text: string, minChars = 10, maxChars = 2000): ValidationResult {
  const trimmed = text.trim();
  const charCount = trimmed.length;
  const words = trimmed ? trimmed.split(/\s+/).filter(Boolean) : [];
  const wordCount = words.length;

  if (charCount === 0) {
    return {
      isValid: false,
      errorMessage: 'Please write or paste some text before clicking Analyse Text.',
      warnings: [],
      charCount: 0,
      wordCount: 0,
    };
  }

  if (charCount < minChars) {
    return {
      isValid: false,
      errorMessage: `Please share a little more text (at least ${minChars} characters) so there is enough context to look at.`,
      warnings: [],
      charCount,
      wordCount,
    };
  }

  const warnings: string[] = [];
  if (charCount > maxChars) {
    warnings.push(
      `Your text is quite long (${charCount} characters). Only the first portion will be reviewed at once.`
    );
  }
  if (wordCount < 4) {
    warnings.push(
      'Very short notes give only a little context, so the result may be less certain.'
    );
  }

  return {
    isValid: true,
    errorMessage: null,
    warnings,
    charCount,
    wordCount,
  };
}

export function preprocessSocialText(
  text: string,
  options: PreprocessingOptions
): PreprocessedResult {
  let normalized = text.normalize('NFKC').replace(CONTROL_CHARS_REGEX, ' ');

  let urlsMasked = 0;
  if (options.maskUrls) {
    normalized = normalized.replace(URL_REGEX, () => {
      urlsMasked += 1;
      return '[URL]';
    });
  }

  let mentionsMasked = 0;
  if (options.anonymizeMentions) {
    normalized = normalized.replace(MENTION_REGEX, () => {
      mentionsMasked += 1;
      return '[USER]';
    });
  }

  if (options.stripHashtagSymbol) {
    normalized = normalized.replace(HASHTAG_REGEX, '$1');
  }

  if (options.normalizeWhitespace) {
    normalized = normalized.replace(MULTI_SPACE_REGEX, ' ').trim();
  }

  if (options.lowercase) {
    normalized = normalized.toLowerCase();
  }

  const words = normalized ? normalized.split(/\s+/).filter(Boolean) : [];

  return {
    rawText: text,
    cleanedText: normalized,
    urlsMasked,
    mentionsMasked,
    charCountRaw: text.length,
    charCountCleaned: normalized.length,
    wordCountCleaned: words.length,
  };
}

/**
 * Deterministic structural WordPiece-style tokenization preview matching BERT/DistilBERT conventions.
 * Exact vocabulary IDs require the Hugging Face AutoTokenizer in Python.
 */
export function previewWordPieceTokens(cleanedText: string, maxSeqLength = 256): TokenPreview {
  if (!cleanedText.trim()) {
    return { tokens: [], subwordCount: 0, truncated: false };
  }

  const rawChunks = cleanedText.match(/\[URL\]|\[USER\]|[A-Za-z0-9]+|[^\sA-Za-z0-9]/g) ?? [];
  const subwords: string[] = ['[CLS]'];

  for (const chunk of rawChunks) {
    if (chunk === '[URL]' || chunk === '[USER]') {
      subwords.push(chunk);
      continue;
    }
    const lower = chunk.toLowerCase();
    if (/^[a-z]{6,}$/.test(lower)) {
      const matchedSuffix = COMMON_SUFFIXES.find(
        (suf) => lower.endsWith(suf) && lower.length - suf.length >= 3
      );
      if (matchedSuffix) {
        const stem = lower.slice(0, lower.length - matchedSuffix.length);
        subwords.push(stem, `##${matchedSuffix}`);
        continue;
      }
    }
    subwords.push(lower);
  }

  subwords.push('[SEP]');
  const truncated = subwords.length > maxSeqLength;
  const bounded = truncated ? [...subwords.slice(0, maxSeqLength - 1), '[SEP]'] : subwords;

  return {
    tokens: bounded,
    subwordCount: bounded.length,
    truncated,
  };
}

export function detectImmediateSelfHarmCues(text: string): boolean {
  return Boolean(text && IMMEDIATE_SELF_HARM_REGEX.test(text));
}

export function detectSupportiveCues(text: string, predictedLabel = ''): boolean {
  if (detectImmediateSelfHarmCues(text)) {
    return true;
  }
  if (text && SUPPORTIVE_CUE_REGEX.test(text)) {
    return true;
  }
  if (predictedLabel) {
    const lowered = predictedLabel.toLowerCase();
    return ['distress', 'depress', 'anxiety', 'stress', 'suicid', 'crisis', 'self-harm', 'risk'].some(
      (kw) => lowered.includes(kw)
    );
  }
  return false;
}

export function buildGroundedExplanationSummary(
  method: ExplanationMethod,
  targetLabel: string,
  confidence: number,
  attributions: TokenAttribution[]
): string {
  const normalized = filterAndNormalizeAttributions(attributions);
  const ranked = [...normalized].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const supporting = ranked
    .filter((a) => a.direction === 'supporting')
    .slice(0, 4)
    .map((a) => `'${a.token}'`);
  const opposing = ranked
    .filter((a) => a.direction === 'opposing')
    .slice(0, 4)
    .map((a) => `'${a.token}'`);

  const supStr = supporting.length > 0 ? supporting.join(', ') : 'none above significance threshold';
  const oppStr = opposing.length > 0 ? opposing.join(', ') : 'none above significance threshold';

  return (
    `Using ${method}, the Transformer model predicted '${targetLabel}' with ${(confidence * 100).toFixed(1)}% ` +
    `softmax confidence. The meaningful words that pushed the model's score most strongly toward '${targetLabel}' were ${supStr}. ` +
    `The meaningful words that pulled the score away from '${targetLabel}' were ${oppStr}. ` +
    `Feature attribution indicates how the Transformer model responded to text features; it does not establish ` +
    `the actual psychological reasons behind a person's writing.`
  );
}
