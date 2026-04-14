import { describe, it, expect, beforeEach } from 'vitest'
import { create } from 'zustand'

interface TestState {
  count: number
  items: string[]
  increment: () => void
  addItem: (item: string) => void
  reset: () => void
}

const useTestStore = create<TestState>((set) => ({
  count: 0,
  items: [],
  increment: () => set((state) => ({ count: state.count + 1 })),
  addItem: (item: string) => set((state) => ({ items: [...state.items, item] })),
  reset: () => set({ count: 0, items: [] }),
}))

describe('Zustand Store', () => {
  beforeEach(() => {
    useTestStore.getState().reset()
  })

  describe('State Updates', () => {
    it('should initialize with default state', () => {
      const state = useTestStore.getState()
      expect(state.count).toBe(0)
      expect(state.items).toEqual([])
    })

    it('should increment count', () => {
      const { increment } = useTestStore.getState()
      increment()
      expect(useTestStore.getState().count).toBe(1)
    })

    it('should add items to array', () => {
      const { addItem } = useTestStore.getState()
      addItem('item1')
      addItem('item2')
      expect(useTestStore.getState().items).toEqual(['item1', 'item2'])
    })

    it('should reset state', () => {
      const { increment, addItem, reset } = useTestStore.getState()
      increment()
      increment()
      addItem('test')
      expect(useTestStore.getState().count).toBe(2)
      expect(useTestStore.getState().items).toHaveLength(1)
      
      reset()
      expect(useTestStore.getState().count).toBe(0)
      expect(useTestStore.getState().items).toEqual([])
    })
  })

  describe('State Persistence', () => {
    it('should maintain state between calls', () => {
      useTestStore.getState().increment()
      useTestStore.getState().increment()
      expect(useTestStore.getState().count).toBe(2)
      
      useTestStore.getState().addItem('persist')
      expect(useTestStore.getState().items).toContain('persist')
    })
  })
})
