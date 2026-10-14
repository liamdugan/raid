// raw data
import axios from 'axios'

export interface Submission {
  date_released: string
  detector_name: string
  contact_info: string
  website?: string | null
  paper_link?: string | null
  huggingface_link?: string | null
  github_link?: string | null
  raid_train_usage?: RaidTrainUsage | null
  submission_id?: string
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
type ScoreIndexItem = string;

async function readIndexedLeaderboardScores(indexPath: string): Promise<Submission[]> {
  const index: ScoreIndexItem[] = await axios.get(indexPath).then(r => r.data)
  return await Promise.all(
    index.map((item) =>
      axios.get(item).then((r) => ({
        ...r.data,
        submission_id: item.replace(/^.*\//, '').replace(/\.json$/, '')
      }))
    )
  )
}

// maintainer-recorded training-data disclosures for submissions whose metadata predates the field
async function readRaidTrainUsageRecord(): Promise<{ [submissionId: string]: string }> {
  try {
    return await axios.get('/data/raid-train-usage.json').then((r) => r.data)
  } catch {
    return {}
  }
}

export async function getLeaderboardScores(): Promise<Submission[]> {
  const [scores, record] = await Promise.all([
    readIndexedLeaderboardScores('/data/submissions/_index.json'),
    readRaidTrainUsageRecord()
  ])
  return scores.map((sub) => {
    const usage = (sub.submission_id && record[sub.submission_id]) || sub.raid_train_usage
    return {
      ...sub,
      raid_train_usage: ALL_RAID_TRAIN_USAGES.includes(usage as RaidTrainUsage)
        ? (usage as RaidTrainUsage)
        : null
    }
  })
}

export async function getSharedTaskScores(): Promise<Submission[]> {
  return await readIndexedLeaderboardScores('/data/shared-task/_index.json')
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

// training-data disclosure (README: Training-data disclosure)
export const ALL_RAID_TRAIN_USAGES = ['none', 'calibration', 'training'] as const
export type RaidTrainUsage = (typeof ALL_RAID_TRAIN_USAGES)[number]
export const RAID_TRAIN_USAGE_LABELS: { [usage in RaidTrainUsage]: string } = {
  none: 'no RAID data',
  calibration: 'calibrated on RAID',
  training: 'trained on RAID'
}
export const RAID_TRAIN_USAGE_DESCRIPTIONS: { [usage in RaidTrainUsage]: string } = {
  none: 'RAID training data was not used at all.',
  calibration:
    'No parameters were learned from RAID training data; it only set a few scalar values (thresholds, constants).',
  training: 'Detector parameters were learned from RAID training data.'
}
export const ALL_RAID_TRAIN_FILTERS = [
  'all',
  'not trained on RAID',
  'trained on RAID',
  'not reported'
] as const

export function matchesRaidTrainFilter(
  datum: Datum,
  filter: (typeof ALL_RAID_TRAIN_FILTERS)[number]
): boolean {
  switch (filter) {
    case 'not trained on RAID':
      return datum.raid_train_usage === 'none' || datum.raid_train_usage === 'calibration'
    case 'trained on RAID':
      return datum.raid_train_usage === 'training'
    case 'not reported':
      return !datum.raid_train_usage
  }
  return true
}

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
