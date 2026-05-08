import React from 'react'
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import Image from "next/image";

interface AgentInfo {
  id: number
  name: string
  description?: string
  status: 'draft' | 'published'
  instructions?: string
}

// Enhanced Markdown detection function: covers broader Markdown features not limited to starting with #
const isLikelyMarkdown = (s: string): boolean => {
  const t = s.trim()
  if (!t) return false
  return (
    t.startsWith('#') || // Heading
    s.includes('```') || // Code block
    s.includes('**') || // Bold
    /(\n|^)\s*(-|\*|\d+\.)\s/.test(s) || // List (unordered/ordered)
    (s.includes('|') && s.includes('---')) || // Table
    /\[[^\]]+\]\([^\)]+\)/.test(s) || // Link [text](url)
    /!\[[^\]]*\]\([^\)]+\)/.test(s) || // Image ![alt](url)
    /(\n|^)\s*>\s/.test(s) || // Blockquote
    /(\n|^)\s*---\s*(\n|$)/.test(s) // Horizontal rule
  )
}

interface MarkdownRendererProps {
  content: string
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
}

const safeUrlTransform = (url: string): string => {
  if (!url) return ''
  if (url.startsWith('file:')) return url
  return defaultUrlTransform(url)
}


export function MarkdownRenderer({ content, className = '', onFileClick, onAgentClick }: MarkdownRendererProps) {
  const components = React.useMemo<Components>(
    () => ({
      a({ node: _node, href, title, children, ...props }) {
        if (href && href.startsWith('file:')) {
          const filePath = href.replace(/^file:/, '')
          const fileNameFromPath = filePath.split('/').pop() || filePath
          const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
            if (onFileClick) {
              e.preventDefault()
              const linkText =
                (typeof children === 'string' ? children : undefined) ??
                (Array.isArray(children)
                  ? children.map((c: any) => (typeof c === 'string' ? c : '')).join('').trim() || undefined
                  : undefined)
              const fallbackTitle = title || linkText || fileNameFromPath
              onFileClick(filePath, fallbackTitle)
            }
          }

          return (
            <a
              href="#"
              data-file-path={filePath}
              className="file-link"
              title={title || undefined}
              onClick={handleClick}
              {...props}
            >
              {children}
            </a>
          )
        }
        return (
          <a href={href || undefined} title={title || undefined} {...props}>
            {children}
          </a>
        )
      },
      img({ node: _node, src, alt, title, ...props }) {
        if (src && src.startsWith('file:')) {
          const filePath = src.replace(/^file:/, '')
          return (
            <Image
              filePath={filePath}
              alt={alt || ''}
              title={title || alt || ''}
              onFileClick={onFileClick}
              className="file-image cursor-pointer"
              {...props}
            />
          )
        }

        return <img src={src || ''} alt={alt || ''} title={title || alt || ''} {...props} />
      }
    }),
    [onFileClick]
  )

  return (
    <div className={`prose prose-invert max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
        urlTransform={safeUrlTransform}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

interface JsonRendererProps {
  data: any
  className?: string
  onFileClick?: (filePath: string, fileName: string) => void
  onAgentClick?: (agentId: string, agentName: string) => void
}

export function JsonRenderer({ data, className = '', onFileClick, onAgentClick }: JsonRendererProps) {
  const [expanded, setExpanded] = React.useState(true)

  if (typeof data === 'string') {
    // Try to parse as JSON first
    try {
      const parsed = JSON.parse(data)
      return <JsonRenderer data={parsed} className={className} onFileClick={onFileClick} onAgentClick={onAgentClick} />
    } catch {
      // If not JSON, try to identify Markdown more comprehensively
      if (isLikelyMarkdown(data)) {
        return <MarkdownRenderer content={data} className={className} onFileClick={onFileClick} onAgentClick={onAgentClick} />
      }
      // Otherwise display as plain text
      return (
        <pre className={`py-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap ${className}`}>
          {data}
        </pre>
      )
    }
  }

  if (typeof data === 'object' && data !== null) {
    // Check if it's a result object with output that might be markdown
    if (data.output && typeof data.output === 'string' && isLikelyMarkdown(data.output.trim())) {
      return (
        <div className={`space-y-3 ${className}`}>
          <div className="bg-muted p-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap">
            <div className="text-green-400 mb-2">✅ Task completed successfully</div>
            <div className="text-gray-400">Goal: {data.goal}</div>
          </div>
          <div className="border-t border-border pt-3">
            <div className="text-sm font-medium text-foreground mb-2">Result:</div>
            <MarkdownRenderer content={data.output} onFileClick={onFileClick} onAgentClick={onAgentClick} />
          </div>
        </div>
      )
    }

    // For other objects, display as formatted JSON
    return (
      <div className={`space-y-2 ${className}`}>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
        >
          {expanded ? '▼' : '▶'} JSON Data
        </button>
        {expanded && (
          <pre className="bg-muted p-3 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    )
  }

  // For other types, display as string
  return (
    <pre className={`bg-muted py-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap ${className}`}>
      {String(data)}
    </pre>
  )
}
