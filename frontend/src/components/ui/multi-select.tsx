"use client"

import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { ChevronDown, X } from "lucide-react"
import { marked } from "marked"
import { useI18n } from "@/contexts/i18n-context"

interface MultiSelectOption {
  value: string
  label: string
  description?: string
  count?: number
}

interface MultiSelectProps {
  values: string[]
  onValuesChange: (values: string[]) => void
  options: MultiSelectOption[]
  placeholder?: string
  className?: string
  creatable?: boolean
  searchable?: boolean
  disabled?: boolean
}

const MarkdownDescription = ({ content }: { content: string }) => {
  const [html, setHtml] = useState<string>("")

  useEffect(() => {
    const parse = async () => {
      try {
        const result = await marked.parse(content)
        setHtml(result)
      } catch (e) {
        setHtml(content)
      }
    }
    parse()
  }, [content])

  if (!html) return null

  return (
    <div
      className="text-xs text-muted-foreground mt-1 ml-2 [&_p]:m-0 [&_p]:leading-normal [&_strong]:text-foreground [&_code]:bg-muted [&_code]:px-1 [&_code]:rounded [&_code]:font-mono [&_ul]:pl-4 [&_ul]:m-0 [&_ol]:pl-4 [&_ol]:m-0"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

export function MultiSelect({ values, onValuesChange, options, placeholder, className, creatable, searchable, disabled }: MultiSelectProps) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [dropdownDirection, setDropdownDirection] = useState<'down' | 'up'>('down')
  const [inputValue, setInputValue] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const buttonRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false)
      }
    }

    if (open) {
      document.addEventListener("mousedown", handleClickOutside)
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [open])

  // Check dropdown direction
  useEffect(() => {
    if (open && buttonRef.current) {
      const buttonRect = buttonRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - buttonRect.bottom - 50
      const spaceAbove = buttonRect.top - 50

      if (spaceBelow < 200 && spaceAbove > spaceBelow) {
        setDropdownDirection('up')
      } else {
        setDropdownDirection('down')
      }
    }
  }, [open])

  // Allow custom values that are not in options
  const selectedOptions = values.map(v => {
    const opt = options.find(o => o.value === v)
    return opt || { value: v, label: v }
  })

  const filteredOptions = options.filter(opt =>
    opt.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    opt.value.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const toggleOpen = () => {
    if (disabled) return
    setOpen(!open)
  }

  const handleSelect = (value: string) => {
    if (disabled) return
    const newValues = values.includes(value)
      ? values.filter(v => v !== value)
      : [...values, value]
    onValuesChange(newValues)
  }

  const handleCreate = () => {
    if (inputValue.trim()) {
      const newValue = inputValue.trim()
      if (!values.includes(newValue)) {
        onValuesChange([...values, newValue])
      }
      setInputValue("")
    }
  }

  const handleRemove = (value: string, e: React.MouseEvent) => {
    if (disabled) return
    e.stopPropagation()
    const newValues = values.filter(v => v !== value)
    onValuesChange(newValues)
  }

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <div
        ref={buttonRef}
        className={cn(
          "flex min-h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          open && "ring-2 ring-ring ring-offset-2",
          disabled && "cursor-not-allowed opacity-50"
        )}
        onClick={toggleOpen}
      >
        <div className="flex items-center gap-2 flex-wrap">
          {selectedOptions.length > 0 ? (
            selectedOptions.map((option) => (
              <div key={option.value} className="flex items-center gap-1 bg-secondary text-secondary-foreground px-2 py-1 rounded-md text-xs">
                {option.label}
                <button
                  type="button"
                  onClick={(e) => handleRemove(option.value, e)}
                  className="hover:text-destructive"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))
          ) : (
            <span className="text-muted-foreground">{placeholder || "Select..."}</span>
          )}
        </div>
        <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform flex-shrink-0", open && "rotate-180")} />
      </div>

      {open && (
        <div className={cn(
          "absolute left-0 right-0 z-[9999] bg-popover text-popover-foreground border border-border rounded-md shadow-lg",
          dropdownDirection === 'down' ? "top-full mt-1" : "bottom-full mb-1"
        )}>
          {searchable && (
             <div className="p-2 border-b">
               <input
                 className="w-full px-2 py-1 text-sm border rounded bg-background"
                 placeholder="Search..."
                 value={searchQuery}
                 onChange={(e) => setSearchQuery(e.target.value)}
                 onClick={(e) => e.stopPropagation()}
               />
             </div>
          )}

          <div className="max-h-60 overflow-auto">
            {filteredOptions.length === 0 && !creatable ? (
              <div className="px-3 py-2 text-sm text-muted-foreground">{t("common.noOptions")}</div>
            ) : (
              filteredOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => handleSelect(option.value)}
                  className={cn(
                    "w-full px-3 py-2 text-sm text-left hover:bg-accent hover:text-accent-foreground",
                    "border-b border-border last:border-b-0",
                    values.includes(option.value) && "bg-accent text-accent-foreground"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{option.label}</span>
                      {values.includes(option.value) && (
                        <div className="w-2 h-2 rounded-full bg-primary" />
                      )}
                    </div>
                    {option.count !== undefined && (
                      <span className="text-xs text-muted-foreground bg-secondary px-2 py-0.5 rounded-full ml-2">
                        {option.count}
                      </span>
                    )}
                  </div>
                  {option.description && (
                    <MarkdownDescription content={option.description} />
                  )}
                </button>
              ))
            )}

            {creatable && (
              <div className="p-2 border-t sticky bottom-0 bg-background" onClick={(e) => e.stopPropagation()}>
                 <div className="flex gap-2">
                   <input
                     className="flex-1 px-2 py-1 text-sm border rounded bg-background"
                     placeholder="Custom..."
                     value={inputValue}
                     onChange={(e) => setInputValue(e.target.value)}
                     onKeyDown={(e) => {
                       if (e.key === 'Enter') {
                         e.preventDefault()
                         handleCreate()
                       }
                     }}
                   />
                   <button
                     type="button"
                     className="px-2 py-1 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90"
                     onClick={handleCreate}
                   >
                     Add
                   </button>
                 </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
