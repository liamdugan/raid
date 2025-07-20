// raw data
export interface Submission {
  date_released: string
  detector_name: string
  contact_info: string
  website?: string | null
  paper_link?: string | null
  huggingface_link?: string | null
  github_link?: string | null
  scores: SubmissionScore[]
}

export interface Datum extends Submission {
}

export interface SubmissionMetric {
  accuracy: number | null
}

export interface SubmissionScore {
  model: string
  domain: string
  decoding: string
  repetition_penalty: string
  attack: string
  accuracy: { [fpr: string]: SubmissionMetric | null }
  auroc: number | null
}

// utils
export function findSplit(
  datum: Datum,
  model: string = 'all',
  domain: string = 'all',
  decoding: string = 'all',
  repetition_penalty: string = 'all',
  attack: string = 'none'
) {
  return datum.scores.find(
    (score) =>
      score.model === model &&
      score.domain === domain &&
      score.decoding === decoding &&
      score.repetition_penalty === repetition_penalty &&
      score.attack === attack
  )
}

export function getMetricValue(
  score: SubmissionScore | undefined,
  metric: typeof ALL_METRICS[number]
): number | null {
  if (score === undefined) return null
  switch (metric) {
    case 'AUROC':
      return score.auroc
    case 'TPR@FPR=5%':
      if (score.accuracy['0.05'] === null) return null
      return score.accuracy['0.05'].accuracy
    case 'TPR@FPR=1%':
      if (score.accuracy['0.01'] === null) return null
      return score.accuracy['0.01'].accuracy
  }
  return null
}

// data
export async function getLeaderboardScores(): Promise<Submission[]> {
  const mod = await import('@/data/all-scores.json')
  return mod.default
}

export async function getSharedTaskScores(): Promise<Submission[]> {
  const mod = await import('@/data/shared-task-scores.json')
  return mod.default
}

// we hardcode this to the set included in RAID for display ordering and speed
export const ALL_GENERATOR_MODELS = [
  'chatgpt',
  'gpt4',
  'gpt3',
  'gpt2',
  'mistral',
  'mistral-chat',
  'cohere',
  'cohere-chat',
  'llama-chat',
  'mpt',
  'mpt-chat'
] as const
export const ALL_DOMAINS = [
  'abstracts',
  'books',
  'news',
  'poetry',
  'recipes',
  'reddit',
  'reviews',
  'wiki'
] as const
export const ALL_DECODINGS = ['greedy', 'sampling'] as const
export const ALL_REPETITION_PENALTIES = ['no', 'yes'] as const
export const ALL_ATTACKS = [
  'whitespace',
  'upper_lower',
  'synonym',
  'perplexity_misspelling',
  'paraphrase',
  'number',
  'insert_paragraphs',
  'homoglyph',
  'article_deletion',
  'alternative_spelling',
  'zero_width_space'
] as const
export const ALL_METRICS = [
  'AUROC',
  'TPR@FPR=5%',
  'TPR@FPR=1%'
] as const

// ===== old dynamic generation =====
// // Array.filter(unique) -> unique elements of that arr, preserving order
// export function unique<T>(value: T, index: number, array: T[]) {
//   return array.indexOf(value) === index
// }
//
// export function noAll(value: string) {
//   return value !== 'all'
// }
//
// export function noNone(value: string) {
//   return value !== 'none'
// }
//
// ASSUMPTION: there is at least one submission and all submissions have the same (model, domain, decoding, repetition_penalty) options
// export const ALL_GENERATOR_MODELS = ALL_SCORES.map(sub => sub.scores)
//   .flat()
//   .map((s) => s.model)
//   .filter(unique)
//   .filter(noAll)
// export const ALL_DOMAINS = ALL_SCORES.map((sub) => sub.scores)
//   .flat()
//   .map((s) => s.domain)
//   .filter(unique)
//   .filter(noAll)
// export const ALL_DECODINGS = ALL_SCORES.map((sub) => sub.scores)
//   .flat()
//   .map((s) => s.decoding)
//   .filter(unique)
//   .filter(noAll)
// export const ALL_REPETITION_PENALTIES = ALL_SCORES.map((sub) => sub.scores)
//   .flat()
//   .map((s) => s.repetition_penalty)
//   .filter(unique)
//   .filter(noAll)
// export const ALL_ATTACKS = ALL_SCORES.map((sub) => sub.scores)
//   .flat()
//   .map((s) => s.attack)
//   .filter(unique)
//   .filter(noAll)
//   .filter(noNone)
