"use client"

import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { ChevronDown, Check } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

export interface SelectOption {
  value: string
  label: string
  description?: string
  isDefault?: boolean
  isSmallFast?: boolean
  isVisual?: boolean
  isCompact?: boolean
  actionIcon?: React.ReactNode
  onAction?: (e: React.MouseEvent) => void
}

interface SelectProps {
  value?: string
  onValueChange: (value: string) => void
  options?: SelectOption[]
  placeholder?: string
  className?: string
  disabled?: boolean
  allowCustom?: boolean
  customPlaceholder?: string
  customButtonText?: string
  onCustomAdd?: (value: string) => void
}

export function Select({
  value,
  onValueChange,
  options = [],
  placeholder,
  className,
  disabled,
  allowCustom,
  customPlaceholder,
  customButtonText,
  onCustomAdd
}: SelectProps) {
  const { t } = useI18n()

  const _customPlaceholder = customPlaceholder || t('common.customPlaceholder')
  const _customButtonText = customButtonText || t('common.add')
  const [open, setOpen] = useState(false)
  const [dropdownDirection, setDropdownDirection] = useState<'down' | 'up'>('down')
  const [customValue, setCustomValue] = useState("")
  const buttonRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Handle clicking outside to close the dropdown
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

  // Check if the dropdown menu should expand up or down
  useEffect(() => {
    if (open && buttonRef.current) {
      const buttonRect = buttonRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - buttonRect.bottom - 50 // 50px is reserved space
      const spaceAbove = buttonRect.top - 50

      // If there is not enough space below and more space above, expand upwards
      if (spaceBelow < 200 && spaceAbove > spaceBelow) {
        setDropdownDirection('up')
      } else {
        setDropdownDirection('down')
      }
    }
  }, [open])

  const selectedOption = options.find(opt => opt.value === value)

  const handleOptionClick = (optionValue: string) => {
    onValueChange(optionValue)
    setOpen(false)
  }

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <div
        ref={buttonRef}
        onClick={() => !disabled && setOpen(!open)}
        className={cn(
          "w-full flex items-center justify-between px-3 py-2 text-sm bg-background border border-input rounded-md min-h-[40px] cursor-pointer",
          "hover:bg-accent hover:text-accent-foreground",
          "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
          disabled && "opacity-50 cursor-not-allowed pointer-events-none"
        )}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {selectedOption ? (
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className="font-medium truncate">{selectedOption.label}</span>
              {(selectedOption.isDefault || selectedOption.isSmallFast || selectedOption.isVisual || selectedOption.isCompact) && (
                <div className="flex gap-1 flex-shrink-0">
                  {selectedOption.isDefault && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary">Default</span>
                  )}
                  {selectedOption.isSmallFast && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-500">Fast</span>
                  )}
                  {selectedOption.isVisual && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-500">Visual</span>
                  )}
                  {selectedOption.isCompact && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/10 text-green-500">Long Context</span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <span className="text-muted-foreground">{placeholder || "Select..."}</span>
          )}
        </div>
        <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform flex-shrink-0", open && "rotate-180")} />
      </div>

      {open && (
        <div className={cn(
          "absolute left-0 right-0 z-[9999] bg-popover border border-border rounded-md shadow-lg flex flex-col",
          dropdownDirection === 'down' ? "top-full mt-1" : "bottom-full mb-1"
        )}>
          <div className="max-h-60 overflow-auto">
            {options.length === 0 ? (
              <div className="px-3 py-2 text-sm text-muted-foreground">{t("common.noOptions")}</div>
            ) : (
              options.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => handleOptionClick(option.value)}
                  className={cn(
                    "w-full px-3 py-2 text-sm text-left hover:bg-accent hover:text-accent-foreground",
                    "border-b border-border last:border-b-0 transition-colors",
                    value === option.value && "bg-accent text-accent-foreground"
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <span className="font-medium truncate">{option.label}</span>
                      {(option.isDefault || option.isSmallFast || option.isVisual || option.isCompact) && (
                        <div className="flex gap-1 flex-shrink-0">
                          {option.isDefault && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary">Default</span>
                          )}
                          {option.isSmallFast && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-500">Fast</span>
                          )}
                          {option.isVisual && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-500">Visual</span>
                          )}
                          {option.isCompact && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/10 text-green-500">Long Context</span>
                          )}
                        </div>
                      )}
                    </div>
                    {option.actionIcon ? (
                      <div
                        className="flex-shrink-0 p-1 hover:bg-muted/50 rounded transition-colors text-muted-foreground hover:text-foreground"
                        onClick={(e) => {
                          e.stopPropagation()
                          if (option.onAction) option.onAction(e)
                        }}
                      >
                        {option.actionIcon}
                      </div>
                    ) : value === option.value && (
                      <Check className="h-4 w-4 text-primary flex-shrink-0" />
                    )}
                  </div>
                  {option.description && (
                    <div className="text-xs text-muted-foreground mt-1 truncate">{option.description}</div>
                  )}
                </button>
              ))
            )}
          </div>
          {allowCustom && (
            <div className="p-2 border-t border-border flex gap-2 bg-muted/10 shrink-0">
              <input
                type="text"
                value={customValue}
                onChange={(e) => setCustomValue(e.target.value)}
                placeholder={_customPlaceholder}
                className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => {
                  e.stopPropagation()
                  if (e.key === 'Enter' && customValue.trim()) {
                    e.preventDefault()
                    if (onCustomAdd) onCustomAdd(customValue.trim())
                    setCustomValue("")
                    setOpen(false)
                  }
                }}
              />
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  if (customValue.trim() && onCustomAdd) {
                    onCustomAdd(customValue.trim())
                    setCustomValue("")
                    setOpen(false)
                  }
                }}
                disabled={!customValue.trim()}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground shadow hover:bg-primary/90 h-8 px-3 shrink-0"
              >
                {_customButtonText}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export {
  Select as SelectRadix,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
} from "./select-radix"
