<script setup lang="ts">
import { ALL_METRICS, type Datum, findSplit, getMetricValue } from '@/table/data'
import { computed } from 'vue'

const props = defineProps<{
  datum: Datum
  model: string
  selectedDomain: string
  selectedDecoding: string
  selectedRepetition: string
  selectedAttack: string
  selectedMetric: typeof ALL_METRICS[number]
  isMaximum: (x: number, attr: (datum: Datum) => number, round?: (x: number) => any) => boolean
}>()

const split = computed(() =>
  findSplit(
    props.datum,
    props.model,
    props.selectedDomain,
    props.selectedDecoding,
    props.selectedRepetition,
    props.selectedAttack
  )
)

const cellColor = computed(() => {
  const acc = getMetricValue(split.value, props.selectedMetric)
  if (acc === undefined || acc === null) return 'transparent'
  return `hsla(${120 * acc}, 100%, 60%, 0.5)`
})
</script>

<template>
  <td
    :class="{
      'has-text-weight-bold': isMaximum(
        getMetricValue(split, selectedMetric) ?? 0,
        (d) =>
          getMetricValue(
            findSplit(
              d,
              model,
              selectedDomain,
              selectedDecoding,
              selectedRepetition,
              selectedAttack
            ),
            selectedMetric
          ) ?? 0
      )
    }"
    :style="{
      'background-color': cellColor
    }"
    class="lb-cell"
  >
    {{ getMetricValue(split, selectedMetric)?.toFixed(3) ?? '--' }}
  </td>
</template>

<style scoped></style>
