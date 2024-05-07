import { TextField } from '@mui/material'
import InputAdornment from '@mui/material/InputAdornment'
import Typography from '@mui/material/Typography'
import React, { useEffect, useRef, useState } from 'react'

const HotkeyFocusTextField = ({
  label,
  value,
  onChange,
  hotkey = '/',
  ...props
}) => {
  const [isFocused, setIsFocused] = useState(false)
  const textFieldRef = useRef(null)
  const handleKeyDown = (event) => {
    if (
      event.key === hotkey &&
      document.activeElement !== textFieldRef.current
    ) {
      event.preventDefault()
      setIsFocused(true)
      textFieldRef.current?.focus()
    }
  }

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [hotkey])

  return (
    <TextField
      {...props}
      label={label}
      value={value}
      onChange={onChange}
      inputRef={textFieldRef}
      autoFocus={isFocused}
      onBlur={() => setIsFocused(false)}
      onFocus={() => setIsFocused(true)}
      InputProps={{
        endAdornment:
          !isFocused && !value ? (
            <InputAdornment position="end">
              <Typography color="textSecondary" style={{ fontSize: 'inherit' }}>
                Type {hotkey} to search
              </Typography>
            </InputAdornment>
          ) : null,
      }}
    />
  )
}

export default HotkeyFocusTextField
