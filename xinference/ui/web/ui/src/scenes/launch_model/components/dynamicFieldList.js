import AddCircle from '@mui/icons-material/AddCircle'
import DeleteIcon from '@mui/icons-material/Delete'
import { Autocomplete, Box, IconButton, Stack, TextField } from '@mui/material'
import React, { useContext } from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../../components/apiContext'

const DynamicFieldList = ({
  name,
  label,
  mode,
  value = [],
  onChange,
  keyPlaceholder = 'key',
  valuePlaceholder = 'value',
  keyOptions = [],
}) => {
  const { t } = useTranslation()
  const { setErrorMsg } = useContext(ApiContext)
  const [errors, setErrors] = React.useState({})

  const handleAdd = () => {
    if (value.length > 0) {
      const lastItem = value[value.length - 1]
      let isIncomplete = false

      if (mode === 'key-value') {
        isIncomplete = Object.values(lastItem).some((val) =>
          typeof val === 'string'
            ? val.trim() === ''
            : val === undefined || val === null
        )
      } else if (mode === 'value') {
        if (typeof lastItem === 'string') {
          isIncomplete = lastItem.trim() === ''
        } else {
          isIncomplete = lastItem === undefined || lastItem === null
        }
      }

      if (isIncomplete) {
        setErrorMsg(t('launchModel.fillCompleteParametersBeforeAdding'))
        return
      }
    }

    const newItem =
      mode === 'key-value'
        ? { [keyPlaceholder]: '', [valuePlaceholder]: '' }
        : ''
    onChange(name, [...value, newItem])
  }

  const handleDelete = (index) => {
    const newItems = [...value]
    newItems.splice(index, 1)
    onChange(name, newItems)
  }

  const handleChange = (index, field, val) => {
    const newItems = [...value]
    newItems[index] = { ...newItems[index], [field]: val }
    if (field === keyPlaceholder) {
      const errorMap = {}
      const keys = new Map()

      newItems.forEach((item, idx) => {
        const key = item[keyPlaceholder]
        if (!key) return

        if (keys.has(key)) {
          const firstIdx = keys.get(key)
          errorMap[firstIdx] = t('launchModel.mustBeUnique', {
            key: keyPlaceholder,
          })
          errorMap[idx] = t('launchModel.mustBeUnique', { key: keyPlaceholder })
        } else {
          keys.set(key, idx)
        }
      })
      setErrors(errorMap)
    }
    onChange(name, newItems)
  }

  const handleChangeValueOnly = (index, val) => {
    const newItems = [...value]
    newItems[index] = val
    onChange(name, newItems)
  }

  return (
    <Box>
      <Stack spacing={2}>
        <Box
          display="flex"
          alignItems="center"
          gap={1}
          paddingLeft={2}
          paddingTop={1}
        >
          <div>{label}</div>
          <IconButton color="primary" onClick={handleAdd}>
            <AddCircle />
          </IconButton>
        </Box>
        {value.map((item, index) => (
          <Box
            key={index}
            display="flex"
            alignItems="start"
            justifyContent="space-between"
            gap={1}
            paddingLeft={2}
          >
            {mode === 'key-value' ? (
              <>
                <Autocomplete
                  style={{ width: '44%' }}
                  freeSolo
                  disablePortal
                  options={keyOptions}
                  groupBy={() => t('components.suggestsCommonParameters')}
                  value={item[keyPlaceholder]}
                  inputValue={item[keyPlaceholder]}
                  onInputChange={(_, newValue) =>
                    handleChange(index, keyPlaceholder, newValue)
                  }
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      className="textHighlight"
                      label={keyPlaceholder}
                      error={!!errors[index]}
                      helperText={errors[index]}
                    />
                  )}
                />
                <TextField
                  className="textHighlight"
                  style={{ width: '44%' }}
                  label={valuePlaceholder}
                  value={item[valuePlaceholder]}
                  onChange={(e) =>
                    handleChange(index, valuePlaceholder, e.target.value)
                  }
                />
              </>
            ) : (
              <TextField
                fullWidth
                className="textHighlight"
                label={valuePlaceholder}
                value={item}
                onChange={(e) => handleChangeValueOnly(index, e.target.value)}
              />
            )}
            <IconButton onClick={() => handleDelete(index)} size="large">
              <DeleteIcon />
            </IconButton>
          </Box>
        ))}
      </Stack>
    </Box>
  )
}

export default DynamicFieldList
