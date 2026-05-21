"use client"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { format } from "date-fns"

interface DateTimePickerProps {
  value?: string
  onChange: (value: string) => void
  placeholder?: string
  label?: string
}

export function DateTimePicker({
  value,
  onChange,
  placeholder = "Select date and time",
  label
}: DateTimePickerProps) {
  const formatDateForDisplay = (dateString: string) => {
    if (!dateString) return ""
    try {
      const date = new Date(dateString)
      return format(date, "yyyy-MM-dd HH:mm")
    } catch {
      return dateString
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value)
  }

  const handleClear = () => {
    onChange("")
  }

  return (
    <div>
      {label && (
        <Label className="text-sm font-medium mb-2 block">{label}</Label>
      )}

      <div className="flex gap-2">
        <Input
          type="datetime-local"
          value={value || ""}
          onChange={handleInputChange}
          placeholder={placeholder}
          className="flex-1"
        />
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="px-3 py-2 text-sm text-destructive hover:text-destructive-foreground border border-destructive/20 hover:border-destructive/40 rounded-md transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {value && (
        <div className="mt-1 text-xs text-muted-foreground">
          Selected time: {formatDateForDisplay(value)}
        </div>
      )}
    </div>
  )
}
