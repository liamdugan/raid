<script setup lang="ts">
import { SortOrder } from './sorters'

// setup
const props = defineProps<{
  index: number | null
  direction: SortOrder
  disallowSortDesc?: boolean
  descFirst?: boolean
}>()
const emit = defineEmits<{
  (e: 'directionChanged', direction: SortOrder): void
}>()

// methods
function cycleSortState() {
  if (props.direction === SortOrder.NONE) {
    emit(
      'directionChanged',
      props.descFirst && !props.disallowSortDesc ? SortOrder.DESC : SortOrder.ASC
    )
  } else if (props.direction === SortOrder.ASC) {
    emit(
      'directionChanged',
      props.descFirst || props.disallowSortDesc ? SortOrder.NONE : SortOrder.DESC
    )
  } else {
    // desc
    emit('directionChanged', props.descFirst ? SortOrder.ASC : SortOrder.NONE)
  }
}
</script>

<template>
  <span class="icon m-0 is-clickable" @click="cycleSortState">
    <font-awesome-icon :icon="['fas', 'sort']" v-if="direction === 0" />
    <font-awesome-icon :icon="['fas', 'sort-up']" v-else-if="direction === 1" />
    <font-awesome-icon :icon="['fas', 'sort-down']" v-else />
    {{ index !== null ? index + 1 : null }}
  </span>
</template>
