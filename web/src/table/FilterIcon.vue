<script setup lang="ts">
import type { FilterOption } from './filters'
import clickOutside from 'click-outside-vue3'
import { defineProps, ref } from 'vue'

// setup
const props = defineProps<{
  options: FilterOption<number>[]
  selected: number[]
}>()
const emit = defineEmits<{
  (e: 'selectionChanged', selectedOptions: number[]): void
}>()
const vClickOutside = clickOutside.directive

// state
const isExpanded = ref(false)

// methods
function onCheckboxChange(event: any, value: number) {
  const selectedOptions = new Set<number>(props.selected)
  if (event.currentTarget.checked) {
    selectedOptions.add(value)
  } else {
    selectedOptions.delete(value)
  }
  emit('selectionChanged', Array.from(selectedOptions))
}

function onClick() {
  isExpanded.value = true
}

function onClickOutside() {
  isExpanded.value = false
}
</script>

<template>
  <div
    class="dropdown"
    :class="{ 'is-active': isExpanded }"
    @click="onClick"
    v-click-outside="onClickOutside"
  >
    <div class="dropdown-trigger">
      <!-- dropdown controller = filter button -->
      <span class="icon m-0 is-clickable" aria-haspopup="true" aria-controls="dropdown-menu">
        <font-awesome-icon :icon="['fas', 'filter']" />
        {{ selected.length ? selected.length : null }}
      </span>
    </div>

    <div class="dropdown-menu" id="dropdown-menu" role="menu">
      <div class="dropdown-content">
        <div class="dropdown-item" v-for="option in options" :key="option.value">
          <label class="checkbox">
            <input
              type="checkbox"
              :checked="selected.includes(option.value)"
              @change="onCheckboxChange($event, option.value)"
            />
            {{ option.label }}
          </label>
        </div>
      </div>
    </div>
  </div>
</template>
