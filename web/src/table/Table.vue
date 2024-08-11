<script setup lang="ts">
import { filters } from './filters'
import { numeric, numericDesc, SortOrder } from './sorters'
import SortIcon from './SortIcon.vue'
import { computed, onMounted, reactive, ref } from 'vue'
import {
  ALL_ATTACKS,
  ALL_DECODINGS,
  ALL_DOMAINS,
  ALL_GENERATOR_MODELS,
  ALL_REPETITION_PENALTIES,
  type Datum,
  findSplit,
  type Submission
} from './data'
import TableCell from '@/table/TableCell.vue'

// setup
const props = defineProps<{
  data: Submission[]
}>()

// state
const pagination = reactive({
  currentPage: 0,
  numPerPage: 50
})
const filterSelections = reactive(new Map<string, Set<any>>())
const sortOrders = reactive(new Map<string, SortOrder>())

// RAID selectors
// const selectedModel = ref('all')
const selectedDomain = ref('all')
const selectedDecoding = ref('all')
const selectedRepetition = ref('all')
const selectedAttack = ref('none')

// computed
const filteredSortedData = computed(() => {
  // RAID specific: select the right score from the settings
  let tempData: Datum[] = [...props.data]
  // filter
  for (const [filterKey, selected] of filterSelections) {
    const filterImpl = filters[filterKey]
    if (!filterImpl) continue
    tempData = tempData.filter(filterImpl.strategy(Array.from(selected)))
  }
  // sort: return first non-zero sort
  return tempData.sort((a, b) => {
    for (const [sorterKey, direction] of sortOrders) {
      let val = 0
      if (direction === SortOrder.ASC) {
        val = numeric(
          (datum) =>
            findSplit(
              datum,
              sorterKey,
              selectedDomain.value,
              selectedDecoding.value,
              selectedRepetition.value,
              selectedAttack.value
            )?.accuracy ?? -999
        )(a, b)
      } else if (direction === SortOrder.DESC) {
        val = numericDesc(
          (datum) =>
            findSplit(
              datum,
              sorterKey,
              selectedDomain.value,
              selectedDecoding.value,
              selectedRepetition.value,
              selectedAttack.value
            )?.accuracy ?? -999
        )(a, b)
      }
      if (val) return val
    }
    return numericDesc(
      (datum) =>
        findSplit(
          datum,
          'all',
          selectedDomain.value,
          selectedDecoding.value,
          selectedRepetition.value,
          selectedAttack.value
        )?.accuracy ?? -999
    )(a, b)
  })
})

const currentPageData = computed(() => {
  return filteredSortedData.value.slice(
    pagination.currentPage * pagination.numPerPage,
    (pagination.currentPage + 1) * pagination.numPerPage
  )
})

const numPages = computed(() => {
  return Math.ceil(filteredSortedData.value.length / pagination.numPerPage)
})

// methods
// filters
function getSelectedFilterOptions(filterKey: string): number[] {
  const selected = filterSelections.get(filterKey)
  if (selected !== undefined) {
    return Array.from(selected)
  }
  return []
}

function onFilterSelectionChange(filterKey: string, selected: number[]) {
  if (!selected.length) {
    filterSelections.delete(filterKey)
  } else {
    filterSelections.set(filterKey, new Set(selected))
  }
  updateQueryParams()
}

// sorters
function getSortIndex(sorterKey: string): number | null {
  const idx = Array.from(sortOrders.keys()).indexOf(sorterKey)
  return idx === -1 ? null : idx
}

function getSortDirection(sorterKey: string): SortOrder {
  return sortOrders.get(sorterKey) ?? SortOrder.NONE
}

function onSortDirectionChange(sorterKey: string, direction: SortOrder) {
  if (direction === SortOrder.NONE) {
    sortOrders.delete(sorterKey)
  } else {
    sortOrders.set(sorterKey, direction)
  }
  updateQueryParams()
}

// query param helpers
function updateQueryParams() {
  const searchParams = new URLSearchParams(window.location.search)

  // build selector query params
  // searchParams.set('model', selectedModel.value)
  searchParams.set('domain', selectedDomain.value)
  searchParams.set('decoding', selectedDecoding.value)
  searchParams.set('repetition', selectedRepetition.value)
  searchParams.set('attack', selectedAttack.value)

  // build sort query param
  searchParams.delete('sort')
  for (const [sorterKey, direction] of sortOrders) {
    searchParams.append('sort', `${sorterKey}:${direction}`)
  }

  // build filter query params
  for (const filterKey of Object.keys(filters)) {
    searchParams.delete(filterKey)
    const oneFilterSelections = filterSelections.get(filterKey)
    if (oneFilterSelections) {
      oneFilterSelections.forEach((v) => searchParams.append(filterKey, v))
    }
  }

  // set query string without reloading
  const newRelativePathQuery = window.location.pathname + '?' + searchParams.toString()
  history.pushState(null, '', newRelativePathQuery)
}

function loadFilterQueryParams() {
  const searchParams = new URLSearchParams(window.location.search)
  for (const [filterKey, filterDef] of Object.entries(filters)) {
    // if the filter is in the query param and valid, set it up
    const filterQuery = searchParams.getAll(filterKey)
    // find the valid options
    let validOptions = []
    for (const queryElem of filterQuery) {
      const matchingOption = filterDef.options.find((option) => option.value == (queryElem ?? 0))
      if (matchingOption) {
        validOptions.push(matchingOption.value)
      }
    }
    // and init the filter
    if (validOptions.length) {
      filterSelections.set(filterKey, new Set(validOptions))
    }
  }
}

function loadSortQueryParams() {
  // if the sorter is in the query param and valid, set it up
  const searchParams = new URLSearchParams(window.location.search)
  const sortQuery = searchParams.getAll('sort')
  for (const sortElem of sortQuery) {
    // ensure direction is valid
    if (!sortElem) continue
    const [sorterKey, direction] = sortElem.split(':', 2)
    if (!(+direction === 1 || +direction === 2)) continue
    // init the sorter
    sortOrders.set(sorterKey, +direction)
  }
}

function loadSelectorQueryParams() {
  const searchParams = new URLSearchParams(window.location.search)
  // selectedModel.value = searchParams.get('model') ?? 'all'
  selectedDomain.value = searchParams.get('domain') ?? 'all'
  selectedDecoding.value = searchParams.get('decoding') ?? 'all'
  selectedRepetition.value = searchParams.get('repetition') ?? 'all'
  selectedAttack.value = searchParams.get('attack') ?? 'none'
}

// other
function clearFilters() {
  // selectedModel.value = 'all'
  selectedDomain.value = 'all'
  selectedDecoding.value = 'all'
  selectedRepetition.value = 'all'
  selectedAttack.value = 'none'
  filterSelections.clear()
  sortOrders.clear()
  updateQueryParams()
}

// hooks
onMounted(() => {
  loadFilterQueryParams()
  loadSortQueryParams()
  loadSelectorQueryParams()
})

// attr max helpers
function isMaximum(
  x: number,
  attr: (datum: Datum) => number,
  round: (x: number) => any = (x) => x.toFixed(3)
): boolean {
  const max = filteredSortedData.value.reduce((acc, v) => Math.max(acc, attr(v)), -Infinity)
  return round(x) === round(max)
}
</script>

<template>
  <!-- filter info -->
  <div class="level mt-2">
    <!-- select splits -->
    <div class="level-left">
      <!-- model -->
      <!--      <div class="level-item">-->
      <!--        <div class="field">-->
      <!--          <label class="label">Generator Model</label>-->
      <!--          <div class="control">-->
      <!--            <div class="select">-->
      <!--              <select v-model="selectedModel" @change="updateQueryParams()">-->
      <!--                <option>all</option>-->
      <!--                <option v-for="model in ALL_GENERATOR_MODELS">{{ model }}</option>-->
      <!--              </select>-->
      <!--            </div>-->
      <!--          </div>-->
      <!--        </div>-->
      <!--      </div>-->
      <!-- domain -->
      <div class="level-item">
        <div class="field filter-control">
          <label class="label">Domain</label>
          <div class="control w-100">
            <div class="select w-100">
              <select class="w-100" v-model="selectedDomain" @change="updateQueryParams()">
                <option>all</option>
                <option v-for="domain in ALL_DOMAINS">{{ domain }}</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      <!-- decoding -->
      <div class="level-item">
        <div class="field filter-control">
          <label class="label">Decoding Strategy</label>
          <div class="control w-100">
            <div class="select w-100">
              <select class="w-100" v-model="selectedDecoding" @change="updateQueryParams()">
                <option>all</option>
                <option v-for="decoding in ALL_DECODINGS">{{ decoding }}</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      <!-- rep -->
      <div class="level-item">
        <div class="field filter-control">
          <label class="label">Repetition Penalty</label>
          <div class="control w-100">
            <div class="select w-100">
              <select class="w-100" v-model="selectedRepetition" @change="updateQueryParams()">
                <option>all</option>
                <option v-for="rep in ALL_REPETITION_PENALTIES">{{ rep }}</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      <!-- attack -->
      <div class="level-item">
        <div class="field filter-control">
          <label class="label">Adversarial Attack</label>
          <div class="control w-100">
            <div class="select w-100">
              <select class="w-100" v-model="selectedAttack" @change="updateQueryParams()">
                <option>none</option>
                <option>all</option>
                <option v-for="atk in ALL_ATTACKS">{{ atk }}</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
    <!-- count, reset -->
    <div class="level-right">
      <p class="level-item">
        {{ filteredSortedData.length }}
        {{ filteredSortedData.length === 1 ? 'entry matches' : 'entries match' }}
        your current filters.
      </p>
      <p class="level-item">
        <button class="button" @click="clearFilters()">Clear Sort &amp; Filters</button>
      </p>
    </div>
  </div>
  <!-- /filter info -->

  <div class="table-container mt-4">
    <table class="table is-striped is-fullwidth is-hoverable">
      <thead>
        <tr>
          <th colspan="2" class="superheader"></th>
          <th :colspan="ALL_GENERATOR_MODELS.length" class="superheader has-text-centered">
            Generator Model
          </th>
        </tr>
        <tr>
          <th>
            <span class="icon-text">
              <span>Detector</span>
            </span>
          </th>
          <th>
            <span class="icon-text">
              <span>Aggregate</span>
              <SortIcon
                class="ml-1"
                :index="getSortIndex('all')"
                :direction="getSortDirection('all')"
                desc-first
                @directionChanged="onSortDirectionChange('all', $event)"
              />
            </span>
          </th>
          <th v-for="gen in ALL_GENERATOR_MODELS">
            <span class="icon-text">
              <span>{{ gen }}</span>
              <SortIcon
                class="ml-1"
                :index="getSortIndex(gen)"
                :direction="getSortDirection(gen)"
                desc-first
                @directionChanged="onSortDirectionChange(gen, $event)"
              />
            </span>
          </th>
        </tr>
      </thead>

      <TransitionGroup tag="tbody" name="lb-rows">
        <tr v-for="datum in currentPageData" :key="datum.detector_name" class="leaderboard-row">
          <td>
            <p>
              {{ datum.detector_name }}
            </p>
            <!-- icons for website, paper, hf, gh -->
            <div class="icon-text">
              <a :href="datum.website" v-if="datum.website" target="_blank">🌐 </a>
              <a :href="datum.paper_link" v-if="datum.paper_link" target="_blank">📄 </a>
              <a :href="datum.huggingface_link" v-if="datum.huggingface_link" target="_blank"
                >🤗
              </a>
              <a
                :href="datum.github_link"
                v-if="datum.github_link"
                target="_blank"
                style="padding-top: 1px"
              >
                <!-- dirty css hack but it works -->
                <span class="icon is-small">
                  <img src="@/assets/github-mark.svg" />
                </span>
              </a>
            </div>
          </td>
          <TableCell
            :datum="datum"
            model="all"
            :selected-domain="selectedDomain"
            :selected-decoding="selectedDecoding"
            :selected-repetition="selectedRepetition"
            :selected-attack="selectedAttack"
            :is-maximum="isMaximum"
          />
          <TableCell
            v-for="gen in ALL_GENERATOR_MODELS"
            :datum="datum"
            :model="gen"
            :selected-domain="selectedDomain"
            :selected-decoding="selectedDecoding"
            :selected-repetition="selectedRepetition"
            :selected-attack="selectedAttack"
            :is-maximum="isMaximum"
          />
        </tr>
      </TransitionGroup>
    </table>

    <div class="level" v-if="numPages > 1">
      <p class="level-item">
        <button
          class="button mr-2"
          v-if="pagination.currentPage > 0"
          @click="pagination.currentPage--"
        >
          <span class="icon is-small">
            <font-awesome-icon :icon="['fas', 'angle-left']" />
          </span>
        </button>
        <span>Page {{ pagination.currentPage + 1 }} / {{ numPages }}</span>
        <button
          class="button ml-2"
          v-if="pagination.currentPage < numPages - 1"
          @click="pagination.currentPage++"
        >
          <span class="icon is-small">
            <font-awesome-icon :icon="['fas', 'angle-right']" />
          </span>
        </button>
      </p>
    </div>
  </div>
</template>

<style scoped>
.table-container {
  min-height: 350px;
}

.filter-control {
  min-width: 12rem;
}

.w-100 {
  width: 100%;
}

.superheader {
  border-bottom: none;
}

.lb-rows-move {
  transition: all 0.5s ease;
}
</style>
