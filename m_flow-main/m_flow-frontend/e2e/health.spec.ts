import { test, expect } from '@playwright/test'

test.describe('Application Health', () => {
  test('should load the home page', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle(/M-Flow/)
  })

  test('should have working navigation', async ({ page }) => {
    await page.goto('/')
    
    // Wait for the app to initialize
    await page.waitForLoadState('networkidle')
    
    // Check that the page is accessible
    const body = page.locator('body')
    await expect(body).toBeVisible()
  })
})
