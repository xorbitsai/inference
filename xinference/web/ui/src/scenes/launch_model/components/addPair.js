import { AddCircle } from '@mui/icons-material'
import DeleteIcon from '@mui/icons-material/Delete'
import { Alert, Box, IconButton, Snackbar, TextField } from '@mui/material'
import React, { useEffect, useState } from 'react'

const AddPair = ({ customData, onGetArr, onJudgeArr }) => {
  const [openSnackbar, setOpenSnackbar] = useState(false)
  const [arr, setArr] = useState([])
  const [arrId, setArrId] = useState(0)
  const [defaultIndex, setDefaultIndex] = useState(-1)
  const [isNotUniqueKey, setIsNotUniqueKey] = useState(false)

  useEffect(() => {
    onGetArr(arr)
  }, [arr])

  const updateArr = (index, type, newValue) => {
    setArr(
      arr.map((pair, subIndex) => {
        if (subIndex === index) {
          return { ...pair, [type]: newValue }
        }
        return pair
      })
    )
    if (type === customData.key) {
      setDefaultIndex(-1)
      setIsNotUniqueKey(false)
      arr.forEach((pair) => {
        if (pair[customData.key] === newValue) {
          setDefaultIndex(index)
          setIsNotUniqueKey(true)
        }
      })
    }
  }

  const handleDeleteArr = (index) => {
    setDefaultIndex(-1)
    setArr(
      arr.filter((_, subIndex) => {
        return index !== subIndex
      })
    )
    onGetArr(arr)
  }

  return (
    <div>
      <Box>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            margin: '20px 0 0 15px',
          }}
        >
          <div>{customData.title}</div>
          <IconButton
            color="primary"
            onClick={() => {
              setArrId(arrId + 1)
              let obj = { id: arrId }
              obj[customData.key] = ''
              obj[customData.value] = ''
              onJudgeArr(arr, [customData.key, customData.value])
                ? setArr([...arr, obj])
                : setOpenSnackbar(true)
            }}
          >
            <AddCircle />
          </IconButton>
        </div>
        <Box>
          {arr.map((item, index) => {
            return (
              <Box key={item.id}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginTop: '10px',
                    marginLeft: '10px',
                  }}
                >
                  <TextField
                    label={customData.key}
                    value={item[customData.key]}
                    onChange={(e) => {
                      updateArr(index, customData.key, e.target.value)
                    }}
                    style={{ width: '44%' }}
                  />
                  <TextField
                    label={customData.value}
                    value={item[customData.value]}
                    onChange={(e) => {
                      updateArr(index, customData.value, e.target.value)
                    }}
                    style={{ width: '44%' }}
                  />
                  <IconButton
                    aria-label="delete"
                    onClick={() => handleDeleteArr(index)}
                    style={{ marginLeft: '10px' }}
                  >
                    <DeleteIcon />
                  </IconButton>
                </div>
                {isNotUniqueKey && defaultIndex === index && (
                  <Alert severity="error">
                    {customData.key} must be unique
                  </Alert>
                )}
              </Box>
            )
          })}
        </Box>
      </Box>
      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={openSnackbar}
        onClose={() => setOpenSnackbar(false)}
        message="Please fill in the complete parameters before adding!!"
        key={'top' + 'center'}
      />
    </div>
  )
}

export default AddPair
