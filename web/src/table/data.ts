import allScores from '@/data/all-scores.json'
import { noAll, noNone, unique } from '@/table/utils'
import sharedTaskScores from '@/data/shared-task-scores.json'

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

export interface Datum extends Submission {}

export interface SubmissionScore {
  model: string
  domain: string
  decoding: string
  repetition_penalty: string
  attack: string
  accuracy: number | null
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

// consts
export const ALL_SCORES: Submission[] = allScores
export const SHARED_TASK_SCORES: Submission[] = sharedTaskScores

// ASSUMPTION: there is at least one submission and all submissions have the same (model, domain, decoding, repetition_penalty) options
// export const ALL_GENERATOR_MODELS = ALL_SCORES.map(sub => sub.scores)
//   .flat()
//   .map((s) => s.model)
//   .filter(unique)
//   .filter(noAll)
// we hardcode this to the set included in RAID for display ordering
export const ALL_GENERATOR_MODELS = [
  "chatgpt",
  "gpt4",
  "gpt3",
  "gpt2",
  "mistral",
  "mistral-chat",
  "cohere",
  "cohere-chat",
  "llama-chat",
  "mpt",
  "mpt-chat",
];
export const ALL_DOMAINS = ALL_SCORES.map(sub => sub.scores)
  .flat()
  .map((s) => s.domain)
  .filter(unique)
  .filter(noAll)
export const ALL_DECODINGS = ALL_SCORES.map(sub => sub.scores)
  .flat()
  .map((s) => s.decoding)
  .filter(unique)
  .filter(noAll)
export const ALL_REPETITION_PENALTIES = ALL_SCORES.map(sub => sub.scores)
  .flat()
  .map((s) => s.repetition_penalty)
  .filter(unique)
  .filter(noAll)
export const ALL_ATTACKS = ALL_SCORES.map(sub => sub.scores)
  .flat()
  .map((s) => s.attack)
  .filter(unique)
  .filter(noAll)
  .filter(noNone)
