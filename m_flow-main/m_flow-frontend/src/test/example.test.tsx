import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

function ExampleComponent() {
  return <div data-testid="example">Hello, Test!</div>
}

describe('Example Test', () => {
  it('should render correctly', () => {
    render(<ExampleComponent />)
    expect(screen.getByTestId('example')).toBeInTheDocument()
    expect(screen.getByText('Hello, Test!')).toBeInTheDocument()
  })
})
