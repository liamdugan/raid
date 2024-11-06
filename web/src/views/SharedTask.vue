<script setup lang="ts">
import Table from '@/table/Table.vue'
import { SHARED_TASK_SCORES } from '@/table/data'
</script>

<template>
  <section class="section">
    <div class="container">
      <h1 class="title">RAID Benchmark: Shared Task Leaderboard</h1>
      <p>
        This is the place for the COLING Shared Task leaderboard. It's cool.
        If the text still says this, Liam hasn't edited it.
      </p>
    </div>
  </section>

  <section class="section">
    <div class="container">
      <Table :data="SHARED_TASK_SCORES" />
    </div>
  </section>
</template>

<style scoped></style>
