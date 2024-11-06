<script setup lang="ts">
</script>

<template>
  <section class="hero">
    <div class="hero-body">
      <div class="container has-text-centered">
        <h1 class="title">
          RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors
        </h1>
        <p class="subtitle authors">
          Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus
          Ludan, Andrew Zhu, Hainiu Xu, Daphne
          Ippolito, Chris Callison-Burch
        </p>

        <div class="is-flex is-flex-direction-row is-justify-content-center logos mt-4">
          <figure class="image is-128x128">
            <img src="@/assets/PennNLP.svg">
          </figure>

          <figure class="image is-128x128">
            <img src="@/assets/uclnlp_logo.png">
          </figure>

          <figure class="image is-128x128">
            <img src="@/assets/KCLNLP.png">
          </figure>

          <figure class="image is-128x128">
            <img src="@/assets/lti-logo_icon_full-color_rgb-300px.png">
          </figure>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="container">
      <div class="content center-justify">
        <p>
          Many commercial and open-source models claim to detect machine-generated text with very
          high accuracy (99% or higher). However, very few of these detectors are evaluated on
          shared benchmark datasets and even when they are, the datasets used for evaluation are
          insufficiently challenging—lacking variations in sampling strategy, adversarial attacks,
          and open-source generative models. In this work we present RAID: the largest and most
          challenging benchmark dataset for machine-generated text detectors. RAID includes over 10
          million generations spanning 12 models, 11 domains, 12 adversarial attacks and 4 decoding
          strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8
          open- and 4 closed-source detectors and find that current detectors are easily fooled by
          adversarial attacks, variations in sampling strategies, repetition penalties, and unseen
          generative models. We release our dataset and tools to encourage further exploration into
          detector robustness at
          <a href="https://github.com/liamdugan/raid" target="_blank"
          >https://github.com/liamdugan/raid</a
          >.
        </p>
      </div>

      <div class="level mt-4">
        <div class="level-item buttons are-medium">
          <a class="button is-primary" href="https://arxiv.org/abs/2405.07940" target="_blank">Paper</a>
          <a class="button is-primary ml-4" href="https://github.com/liamdugan/raid" target="_blank">GitHub</a>
          <RouterLink to="/leaderboard" class="button is-primary ml-4">Leaderboard</RouterLink>
          <RouterLink to="/shared-task" class="button is-primary ml-4">COLING Shared Task</RouterLink>
<!--          <RouterLink to="/faq" class="button is-primary ml-4">FAQ</RouterLink>-->
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.logos {
  gap: 16px;
}

.center-justify {
  text-align: justify;
  text-align-last: center;
}
</style>
