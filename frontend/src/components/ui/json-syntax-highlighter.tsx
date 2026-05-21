"use client"

import hljs from "highlight.js"
import { useTheme } from 'next-themes';
import { useEffect, useRef } from "react"

interface JSONSyntaxHighlighterProps {
  data: unknown
  className?: string
}

export function JSONSyntaxHighlighter({ data, className = "" }: JSONSyntaxHighlighterProps) {
  const codeRef = useRef<HTMLElement>(null)
  const { theme } = useTheme()

  useEffect(() => {
    // Dynamically load syntax highlighting styles suitable for the current theme
    const loadThemeStyle = async () => {
      // Remove previously loaded styles
      const existingStyles = document.querySelectorAll('link[data-highlight-theme]')
      existingStyles.forEach(style => style.remove())

      // Select appropriate style based on theme
      const resolvedTheme = theme === 'light' ? 'github' : 'github-dark'
      const link = document.createElement('link')
      link.rel = 'stylesheet'
      link.href = `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/${resolvedTheme}.min.css`
      link.setAttribute('data-highlight-theme', resolvedTheme)
      document.head.appendChild(link)
    }

    loadThemeStyle()
  }, [theme])

  useEffect(() => {
    if (codeRef.current) {
      hljs.highlightElement(codeRef.current)
    }
  }, [data])

  const formatJSON = (data: unknown): string => {
    try {
      return JSON.stringify(data, null, 2)
    } catch {
      return String(data)
    }
  }

  return (
    <pre className={`bg-muted p-4 rounded-lg overflow-x-auto ${className}`}>
      <code ref={codeRef} className="language-json text-sm">
        {formatJSON(data)}
      </code>
    </pre>
  )
}
