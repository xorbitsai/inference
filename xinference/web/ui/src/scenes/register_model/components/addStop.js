import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import { Alert, Button, TextField } from '@mui/material'
import React, { useEffect, useState } from 'react'

const regex = /^[1-9]\d*$/

const AddStop = ({
  label,
  onGetData,
  arrItemType,
  formData,
  onGetError,
  helperText,
}) => {
  const [dataArr, setDataArr] = useState(formData?.length ? formData : [''])
  const arr = []

  useEffect(() => {
    if (arrItemType === 'number') {
      const newDataArr = dataArr.map((item) => {
        if (item && regex.test(item)) {
          arr.push('true')
          return Number(item)
        }
        if (item && !regex.test(item)) arr.push('false')
        return item
      })
      onGetError(arr)
      onGetData(newDataArr)
    } else {
      onGetData(dataArr)
    }
  }, [dataArr])

  const handleChange = (value, index) => {
    const arr = [...dataArr]
    arr[index] = value
    setDataArr([...arr])
  }

  const handleAdd = () => {
    if (dataArr[dataArr.length - 1]) {
      setDataArr([...dataArr, ''])
    }
  }

  const handleDelete = (index) => {
    setDataArr(dataArr.filter((_, subIndex) => index !== subIndex))
  }

  const handleShowAlert = (item) => {
    return item !== '' && !regex.test(item) && arrItemType === 'number'
  }

  return (
    <>
      <div>
        <div
          style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}
        >
          <label style={{ width: '100px' }}>{label}</label>
          <Button
            variant="contained"
            size="small"
            endIcon={<AddIcon />}
            className="addBtn"
            onClick={handleAdd}
          >
            more
          </Button>
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            marginInline: 50,
            padding: 20,
            backgroundColor: '#eee',
            borderRadius: 10,
          }}
        >
          {dataArr.map((item, index) => (
            <div key={index}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <TextField
                  value={item}
                  onChange={(e) => handleChange(e.target.value, index)}
                  label={helperText}
                  size="small"
                  style={{ width: '100%' }}
                />
                {dataArr.length > 1 && (
                  <DeleteIcon
                    onClick={() => handleDelete(index)}
                    style={{ cursor: 'pointer', color: '#1976d2' }}
                  />
                )}
              </div>

              {handleShowAlert(item) && (
                <Alert severity="error">
                  Please enter an integer greater than 0.
                </Alert>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

export default AddStop
