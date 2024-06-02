import { type Datum } from './data'

// helpers
export type Sorter = (a: Datum, b: Datum, ...args: string[]) => number

const inverse = (f: Sorter) => (a: Datum, b: Datum, ...args: string[]) => -f(a, b, ...args)

export const enum SortOrder {
  NONE = 0,
  ASC = 1,
  DESC = 2
}

// ==== sorter impls ====
export const numeric: (getter: (datum: Datum) => number) => Sorter = (getter) => (a, b) =>
  getter(a) - getter(b)
export const numericDesc: (getter: (datum: Datum) => number) => Sorter = (key) => inverse(numeric(key))
