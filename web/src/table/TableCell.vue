<script setup lang="ts">
import { type Datum, findSplit } from '@/table/data'
import { computed } from 'vue'

const props = defineProps<{
  datum: Datum
  model: string
  selectedDomain: string
  selectedDecoding: string
  selectedRepetition: string
  selectedAttack: string
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
  const acc = split.value?.accuracy
  if (acc === undefined || acc === null) return 'transparent'
  return `hsla(${120 * acc}, 100%, 60%, 0.5)`
})
</script>

<template>
  <td
    :class="{
      'has-text-weight-bold': isMaximum(
        split?.accuracy ?? 0,
        (d) =>
          findSplit(d, model, selectedDomain, selectedDecoding, selectedRepetition, selectedAttack)
            ?.accuracy ?? 0
      )
    }"
    :style="{
      'background-color': cellColor
    }"
    class="lb-cell"
  >
    {{ split?.accuracy?.toFixed(3) ?? '--' }}
  </td>
</template>

<style scoped></style>
