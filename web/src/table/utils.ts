// Array.filter(unique) -> unique elements of that arr, preserving order
export function unique<T>(value: T, index: number, array: T[]) {
  return array.indexOf(value) === index
}

export function noAll(value: string) {
  return value !== 'all'
}

export function noNone(value: string) {
  return value !== 'none'
}
