import { FormControl, InputLabel, MenuItem, Select } from '@mui/material'

const SelectField = ({
  label,
  labelId,
  name,
  value,
  onChange,
  options = [],
  disabled = false,
  required = false,
}) => (
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
    >
      {options.map((item) => (
        <MenuItem
          key={item.value || item}
          value={item.value || item}
          disabled={item.disabled}
        >
          {item.label || item}
        </MenuItem>
      ))}
    </Select>
  </FormControl>
)

export default SelectField
