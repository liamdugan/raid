import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

import { library } from '@fortawesome/fontawesome-svg-core'
import {
  faAngleLeft,
  faAngleRight,
  faFilter,
  faSort,
  faSortDown,
  faSortUp
} from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

// ==== fontawesome ====
library.add(faAngleLeft, faAngleRight, faSort, faSortUp, faSortDown, faFilter)

createApp(App).use(router).component('font-awesome-icon', FontAwesomeIcon).mount('#app')
