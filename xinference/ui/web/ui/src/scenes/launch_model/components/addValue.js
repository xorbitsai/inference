import { AddCircle } from '@mui/icons-material'
import DeleteIcon from '@mui/icons-material/Delete'
import { Box, IconButton, Snackbar, TextField } from '@mui/material'
import React, { useEffect, useState } from 'react'

const AddValue = ({ customData, pairData, onGetArr, onJudgeArr }) => {
  const [openSnackbar, setOpenSnackbar] = useState(false)
  const [arr, setArr] = useState([])
  const [arrId, setArrId] = useState(0)

  useEffect(() => {
    onGetArr(arr)
  }, [arr])

  useEffect(() => {
    if (!Array.isArray(pairData)) return
    const dataArr = []
    pairData.forEach((item, index) => {
      dataArr.push({
        id: index,
        [customData.value]: item,
      })
    })
    setArrId(pairData.length)
    setArr(dataArr)
  }, [pairData])

  const updateArr = (index, type, newValue) => {
    setArr(
      arr.map((pair, subIndex) => {
        if (subIndex === index) {
          return { ...pair, [type]: newValue }
        }
        return pair
      })
    )
  }

  const handleDeleteArr = (index) => {
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
              obj[customData.value] = ''
              onJudgeArr(arr, [customData.value])
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
                    className="textHighlight"
                    label={customData.value}
                    value={item[customData.value]}
                    onChange={(e) => {
                      updateArr(index, customData.value, e.target.value)
                    }}
                    fullWidth
                  />
                  <IconButton
                    aria-label="delete"
                    onClick={() => handleDeleteArr(index)}
                    style={{ marginLeft: '10px' }}
                  >
                    <DeleteIcon />
                  </IconButton>
                </div>
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

export default AddValue
