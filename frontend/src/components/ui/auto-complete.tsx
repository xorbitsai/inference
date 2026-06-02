"use client"

import * as React from "react"
import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"

import { Check, ChevronDown, X } from "lucide-react"

import { cn } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"

export interface AutoCompleteOption {
  label: string
  value: string
  description?: string
}

interface AutoCompleteProps {
  value?: string
  onChange?: (value?: string) => void

  options?: AutoCompleteOption[]

  placeholder?: string
  className?: string

  disabled?: boolean
  error?: boolean

  allowClear?: boolean

  /**
   * Whether to allow entering values not in options
   */
  allowCustomValue?: boolean

  /**
   * Text to display when there is no data
   */
  emptyText?: string
}

export function AutoComplete({
  value,
  onChange,

  options = [],

  placeholder,
  className,

  disabled,
  error,

  allowClear = true,

  allowCustomValue = true,

  emptyText,
}: AutoCompleteProps) {
  const { t } = useI18n()

  const containerRef =
    useRef<HTMLDivElement>(null)

  const inputRef =
    useRef<HTMLInputElement>(null)

  const [open, setOpen] = useState(false)

  const [inputValue, setInputValue] =
    useState("")

  /**
   * Whether the user is currently typing
   */
  const [typing, setTyping] = useState(false)

  const selectedOption = useMemo(() => {
    return options.find(
      (option) => option.value === value
    )
  }, [options, value])

  /**
   * Sync input when value changes
   */
  useEffect(() => {
    if (!typing) {
      setInputValue(selectedOption?.label || value || "")
    }
  }, [selectedOption, value, typing])

  const handleBlurBehavior = React.useCallback(() => {
    setOpen(false)
    setTyping(false)

    /**
     * Allow free text input
     */
    if (allowCustomValue) {
      onChange?.(inputValue || undefined)
      return
    }

    /**
     * Do not allow free text input
     * Restore the selected value
     */
    setInputValue(
      selectedOption?.label || ""
    )
  }, [
    allowCustomValue,
    inputValue,
    onChange,
    selectedOption?.label,
  ])

  /**
   * Close when clicking outside
   */
  useEffect(() => {
    const handleClickOutside = (
      event: MouseEvent
    ) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(
          event.target as Node
        )
      ) {
        if (
          inputRef.current &&
          document.activeElement ===
          inputRef.current
        ) {
          inputRef.current.blur()
        } else {
          handleBlurBehavior()
        }
      }
    }

    if (open) {
      document.addEventListener(
        "mousedown",
        handleClickOutside
      )
    }

    return () => {
      document.removeEventListener(
        "mousedown",
        handleClickOutside
      )
    }
  }, [handleBlurBehavior, open])

  const filteredOptions = useMemo(() => {
    const keyword =
      inputValue.trim().toLowerCase()

    if (!keyword) {
      return options
    }

    return options.filter((option) => {
      return (
        option.label
          .toLowerCase()
          .includes(keyword) ||
        option.value
          .toLowerCase()
          .includes(keyword) ||
        option.description
          ?.toLowerCase()
          .includes(keyword)
      )
    })
  }, [options, inputValue])

  const handleSelect = (
    option: AutoCompleteOption
  ) => {
    onChange?.(option.value)

    setInputValue(option.label)

    setTyping(false)
    setOpen(false)
  }

  const handleClear = (
    e: React.MouseEvent
  ) => {
    e.stopPropagation()

    if (disabled) return

    setInputValue("")
    setTyping(false)

    onChange?.(undefined)

    inputRef.current?.focus()
  }

  return (
    <div
      ref={containerRef}
      className={cn("relative", className)}
    >
      <div
        className={cn(
          "border-input flex h-9 w-full items-center rounded-md border bg-transparent px-3 py-1 text-sm outline-none transition-all",

          "focus-within:border-ring focus-within:ring-ring/50 focus-within:ring-[3px]",

          error &&
            "border-destructive focus-within:border-destructive focus-within:ring-destructive/40",

          disabled &&
            "cursor-not-allowed opacity-50"
        )}
        onClick={() => {
          if (disabled) return

          inputRef.current?.focus()
        }}
      >
        <input
          ref={inputRef}
          value={inputValue}
          disabled={disabled}
          placeholder={placeholder}
          className={cn(
            "flex-1 bg-transparent outline-none placeholder:text-muted-foreground",
            disabled &&
              "cursor-not-allowed"
          )}
          onFocus={() => {
            if (disabled) return

            setOpen(true)
          }}
          onBlur={handleBlurBehavior}
          onChange={(e) => {
            if (disabled) return

            setTyping(true)
            setInputValue(e.target.value)

            if (!open) {
              setOpen(true)
            }
          }}
          onKeyDown={(e) => {
            if (disabled) return
            console.log(e.key, 'key')
            /**
             * Tab
             */
            if (e.key === "Tab") {
              setOpen(false)
            }

            /**
             * Enter:
             * allowCustomValue=true submit value
             */
            if (
              e.key === "Enter" &&
              allowCustomValue
            ) {
              e.preventDefault()

              inputRef.current?.blur()
            }

            /**
             * ESC
             */
            if (e.key === "Escape") {
              setOpen(false)
              inputRef.current?.blur()
            }
          }}
        />

        {allowClear && value ? (
          <button
            type="button"
            disabled={disabled}
            onClick={handleClear}
            className={cn(
              "ml-2 flex h-4 w-4 shrink-0 items-center justify-center rounded-sm text-muted-foreground transition-colors",

              !disabled &&
                "hover:bg-muted hover:text-foreground"
            )}
          >
            <X className="h-4 w-4" />
          </button>
        ) : (
          <ChevronDown
            className={cn(
              "ml-2 h-4 w-4 shrink-0 text-muted-foreground transition-transform",
              open && "rotate-180"
            )}
          />
        )}
      </div>

      {open && (
        <div className="absolute top-full left-0 right-0 z-[9999] mt-1 overflow-hidden rounded-md border border-border bg-popover shadow-lg">
          <div className="max-h-60 overflow-auto">
            {filteredOptions.length === 0 ? (
              <div className="py-10 text-center text-sm text-muted-foreground">
                {emptyText ||
                  t("common.noOptions")}
              </div>
            ) : (
              filteredOptions.map((option) => {
                const active =
                  value === option.value

                return (
                  <button
                    key={option.value}
                    type="button"
                    className={cn(
                      "flex w-full items-start justify-between gap-2 border-b border-border px-3 py-2 text-left text-sm transition-colors last:border-b-0",

                      "hover:bg-accent hover:text-accent-foreground",

                      active &&
                        "bg-accent text-accent-foreground"
                    )}
                    onMouseDown={(e) => {
                      e.preventDefault()
                    }}
                    onClick={() =>
                      handleSelect(option)
                    }
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-medium">
                        {option.label}
                      </div>

                      {option.description && (
                        <div className="mt-1 truncate text-xs text-muted-foreground">
                          {option.description}
                        </div>
                      )}
                    </div>

                    {active && (
                      <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    )}
                  </button>
                )
              })
            )}
          </div>
        </div>
      )}
    </div>
  )
}
