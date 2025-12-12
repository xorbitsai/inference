import { FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import { useEffect, useRef, useState } from 'react'

const SelectField = ({
  label,
  labelId,
  name,
  value,
  onChange,
  options = [],
  disabled = false,
  required = false,
}) => {
  const wrapperRef = useRef(null)
  const [menuWidth, setMenuWidth] = useState(0)

  useEffect(() => {
    if (wrapperRef.current) {
      setMenuWidth(wrapperRef.current.offsetWidth)
    }
  }, [])

  return (
    <div ref={wrapperRef}>
      <FormControl
        variant="outlined"
        margin="normal"
        disabled={disabled}
        required={required}
        fullWidth
      >
        <InputLabel id={labelId}>{label}</InputLabel>
        <Select
          labelId={labelId}
          name={name}
          value={value}
          onChange={onChange}
          label={label}
          className="textHighlight"
          MenuProps={{
            PaperProps: {
              sx: {
                width: menuWidth,
                maxWidth: menuWidth,
              },
            },
          }}
        >
          {options.map((item) => (
            <MenuItem
              key={item.value || item}
              value={item.value || item}
              disabled={item.disabled}
              sx={{
                whiteSpace: 'normal',
              }}
            >
              {item.label || item}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </div>
  )
}

export default SelectField
