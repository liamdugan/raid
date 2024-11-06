import Home from '@/views/Home.vue'
import NotFound from '@/views/NotFound.vue'
import { nextTick } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import AboutUs from '@/views/AboutUs.vue'
import FAQ from '@/views/FAQ.vue'

const DEFAULT_TITLE = 'RAID Benchmark'

const routes = [
  { path: '/', name: 'Home', component: Home },
  {
    path: '/leaderboard',
    name: 'Leaderboard',
    component: () => import(/* webpackChunkName: "leaderboard" */ '@/views/Leaderboard.vue'),
    meta: { title: 'Leaderboard - RAID Benchmark' }
  },
  {
    path: '/shared-task',
    name: 'Shared Task Leaderboard',
    component: () => import(/* webpackChunkName: "sharedtask" */ '@/views/SharedTask.vue'),
    meta: { title: 'COLING Shared Task - RAID Benchmark' }
  },
  { path: '/about', name: 'About Us', component: AboutUs },
  { path: '/faq', name: 'FAQ', component: FAQ },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: NotFound,
    meta: { title: 'Not Found - RAID Benchmark' }
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

router.afterEach((to) => {
  nextTick(() => {
    document.title = (to.meta.title as string) || DEFAULT_TITLE
  })
})

export default router
