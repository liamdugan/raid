import type { Datum } from './data'

export interface FilterOption<T> {
  label: string
  value: T
}

export type Filter = (datum: Datum) => boolean

export interface FilterParams<T> {
  options: FilterOption<T>[]
  strategy: (values: T[]) => Filter
}

export const filters: { [id: string]: FilterParams<any> } = {
  // modelType: {
  //   options: ALL_DETECTOR_CLASSES.map((c) => ({ label: c, value: c })),
  //   strategy(values: any[]): (datum: Datum) => boolean {
  //     return (datum: Datum) => values.includes(datum.detector_class)
  //   }
  // }
}
