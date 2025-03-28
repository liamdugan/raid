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
        These are the results for the
        <a href="https://genai-content-detection.gitlab.io/sharedtasks" target="_blank">
          COLING 2025 Cross-Domain Machine-Generated Text Detection Shared Task.
        </a>
        <br></br>
        Please consult our
        <a href="https://arxiv.org/abs/2501.08913" target="_blank">
          ArXiv Paper
        </a>
        for more detailed analysis of the results.
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
