import {
  Checkbox,
  FormControl,
  FormHelperText,
  InputLabel,
  ListItemText,
  MenuItem,
  Select,
} from '@mui/material'
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
  multiple = false,
  error = false,
  helperText = '',
}) => {
  const wrapperRef = useRef(null)
  const [menuWidth, setMenuWidth] = useState(0)

  const normalizedValue = multiple
    ? Array.isArray(value)
      ? value
      : value
      ? [value]
      : []
    : value

  const optionLabelMap = new Map(
    options.map((item) => [item.value ?? item, item.label ?? item])
  )

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
        error={error}
      >
        <InputLabel id={labelId}>{label}</InputLabel>
        <Select
          labelId={labelId}
          name={name}
          multiple={multiple}
          value={normalizedValue}
          onChange={onChange}
          label={label}
          className="textHighlight"
          renderValue={
            multiple
              ? (selected) =>
                  selected
                    .map((item) => optionLabelMap.get(item) ?? item)
                    .join(', ')
              : undefined
          }
          MenuProps={{
            PaperProps: {
              sx: {
                width: menuWidth,
                maxWidth: menuWidth,
              },
            },
          }}
        >
          {options.map((item) => {
            const itemValue = item.value ?? item
            const itemLabel = item.label ?? item
            return (
              <MenuItem
                key={itemValue}
                value={itemValue}
                disabled={item.disabled}
                sx={{
                  whiteSpace: 'normal',
                }}
              >
                {multiple ? (
                  <>
                    <Checkbox checked={normalizedValue.includes(itemValue)} />
                    <ListItemText primary={itemLabel} />
                  </>
                ) : (
                  itemLabel
                )}
              </MenuItem>
            )
          })}
        </Select>
        {helperText ? <FormHelperText>{helperText}</FormHelperText> : null}
      </FormControl>
    </div>
  )
}

export default SelectField
