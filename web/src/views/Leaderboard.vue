<script setup lang="ts">
import Table from '@/table/Table.vue'
import { ALL_SCORES } from '@/table/data'
</script>

<template>
  <section class="section">
    <div class="container">
      <h1 class="title">RAID Benchmark Leaderboard</h1>
      <p>
        These leaderboards contain the test-set scores of various detector models. To submit your own model's
        predictions to the leaderboards, see
        <a href="https://github.com/liamdugan/raid#leaderboard-submission" target="_blank">
          Leaderboard Evaluation.
        </a>
      </p>
    </div>
  </section>

  <section class="section">
    <div class="container">
      <Table :data="ALL_SCORES" />
    </div>
  </section>
</template>

<style scoped></style>
