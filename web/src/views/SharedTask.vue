<script setup lang="ts">
import Table from '@/table/Table.vue'
import { onMounted, ref, shallowRef } from 'vue'
import { getSharedTaskScores, type Submission } from '@/table/data'

const scores = shallowRef<Submission[]>([])
const isLoading = ref(true)

onMounted(async () => {
  scores.value = await getSharedTaskScores()
  isLoading.value = false
})
</script>

<template>
  <section class="section">
    <div class="container">
      <h1 class="title">RAID Benchmark: Shared Task Leaderboard</h1>
      <p>
        This is the place for the COLING Shared Task leaderboard. It's cool. If the text still says
        this, Liam hasn't edited it.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="container">
      <progress class="progress" v-if="isLoading" max="100">Loading</progress>
      <Table :data="scores" />
    </div>
  </section>
</template>

<style scoped></style>
