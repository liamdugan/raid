<template>
  <nav class="navbar" role="navigation" aria-label="main navigation">
    <div class="container">
      <div class="navbar-brand">
        <a
          role="button"
          class="navbar-burger"
          aria-label="menu"
          aria-expanded="false"
          data-target="navbarBasicExample"
        >
          <span aria-hidden="true"></span>
          <span aria-hidden="true"></span>
          <span aria-hidden="true"></span>
        </a>
      </div>

      <div id="navbarBasicExample" class="navbar-menu">
        <div class="navbar-start">
          <RouterLink to="/" class="navbar-item"> Home</RouterLink>
          <RouterLink to="/leaderboard" class="navbar-item"> Leaderboard</RouterLink>
          <!--          <RouterLink to="/about" class="navbar-item"> About Us </RouterLink>-->
          <!--          <RouterLink to="/faq" class="navbar-item"> FAQ </RouterLink>-->
        </div>
      </div>
    </div>
  </nav>

  <router-view />
</template>

<style lang="scss">
@import './global.scss';
</style>

<script setup lang="ts">
</script>
