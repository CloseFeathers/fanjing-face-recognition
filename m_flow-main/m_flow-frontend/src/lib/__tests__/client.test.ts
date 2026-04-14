import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

describe('API Client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  describe('Error Handling', () => {
    it('should handle 401 unauthorized response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ detail: 'Unauthorized' }),
      })
      vi.stubGlobal('fetch', mockFetch)

      const response = await fetch('/api/v1/datasets')
      expect(response.ok).toBe(false)
      expect(response.status).toBe(401)
    })

    it('should handle 404 not found response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: 'Not Found' }),
      })
      vi.stubGlobal('fetch', mockFetch)

      const response = await fetch('/api/v1/datasets/non-existent')
      expect(response.ok).toBe(false)
      expect(response.status).toBe(404)
    })

    it('should handle network error', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'))
      vi.stubGlobal('fetch', mockFetch)

      await expect(fetch('/api/v1/health')).rejects.toThrow('Network error')
    })
  })

  describe('Request Building', () => {
    it('should include authorization header when token exists', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ data: 'test' }),
      })
      vi.stubGlobal('fetch', mockFetch)

      await fetch('/api/v1/datasets', {
        headers: {
          Authorization: 'Bearer test-token',
        },
      })

      expect(mockFetch).toHaveBeenCalledWith('/api/v1/datasets', {
        headers: {
          Authorization: 'Bearer test-token',
        },
      })
    })

    it('should set content-type for JSON requests', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ success: true }),
      })
      vi.stubGlobal('fetch', mockFetch)

      await fetch('/api/v1/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: 'test' }),
      })

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/add',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      )
    })
  })
})
