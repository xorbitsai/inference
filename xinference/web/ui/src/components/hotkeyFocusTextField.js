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
  const textFieldRef = useRef(null) // 创建一个ref来引用TextField
  const handleKeyDown = (event) => {
    // 检测到按下'/'键
    if (
      event.key === hotkey &&
      document.activeElement !== textFieldRef.current
    ) {
      event.preventDefault() // 阻止默认行为
      setIsFocused(true) // 更新状态以聚焦TextField
      // 检查并确保TextField已经渲染
      textFieldRef.current?.focus() // 使用ref聚焦TextField
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
