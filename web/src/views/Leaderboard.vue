<script setup lang="ts">
import Table from '@/table/Table.vue'
import { onMounted, ref, shallowRef } from 'vue'
import { getLeaderboardScores, type Submission } from '@/table/data'

const scores = shallowRef<Submission[]>([])
const isLoading = ref(true)

onMounted(async () => {
  scores.value = await getLeaderboardScores()
  isLoading.value = false
})
</script>

<template>
  <section class="section">
    <div class="container">
      <h1 class="title">RAID Benchmark Leaderboard</h1>
      <p>
        These leaderboards contain the test-set scores of various detector models. To submit your
        own model's predictions to the leaderboards, see
        <a href="https://github.com/liamdugan/raid#leaderboard-submission" target="_blank">
          Leaderboard Evaluation.
        </a>
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
