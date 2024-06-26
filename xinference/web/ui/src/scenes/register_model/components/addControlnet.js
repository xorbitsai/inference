import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import {
  Box,
  Button,
  FormControlLabel,
  Radio,
  RadioGroup,
  TextField,
  Tooltip,
} from '@mui/material'
import React, { useEffect, useState } from 'react'

const AddControlnet = ({
  controlnetDataArr,
  onGetControlnetArr,
  scrollRef,
}) => {
  const [count, setCount] = useState(0)
  const [controlnetArr, setControlnetArr] = useState([])
  const [isAdd, setIsAdd] = useState(false)

  useEffect(() => {
    if (controlnetDataArr && controlnetDataArr.length) {
      const dataArr = controlnetDataArr.map((item) => {
        setCount(count + 1)
        item.id = count
        return item
      })
      setControlnetArr(dataArr)
    }
  }, [])

  useEffect(() => {
    const arr = controlnetArr.map((item) => {
      const { model_name: name, model_uri: uri, model_family } = item
      return {
        model_name: name,
        model_uri: uri,
        model_family,
      }
    })
    onGetControlnetArr(arr)
    isAdd && handleScrollBottom()
    setIsAdd(false)
  }, [controlnetArr])

  const handleAddControlnet = () => {
    setCount(count + 1)
    const item = {
      id: count,
      model_name: 'custom-controlnet',
      model_uri: '/path/to/controlnet-model',
      model_family: 'controlnet',
    }
    setControlnetArr([...controlnetArr, item])
    setIsAdd(true)
  }

  const handleUpdateSpecsArr = (index, type, newValue) => {
    setControlnetArr(
      controlnetArr.map((item, subIndex) => {
        if (subIndex === index) {
          return { ...item, [type]: newValue }
        }
        return item
      })
    )
  }

  const handleDeleteControlnet = (index) => {
    setControlnetArr(controlnetArr.filter((_, subIndex) => index !== subIndex))
  }

  const handleScrollBottom = () => {
    scrollRef.current.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }

  return (
    <>
      <div>
        <label style={{ marginBottom: '20px' }}>Controlnet</label>
        <Button
          variant="contained"
          size="small"
          endIcon={<AddIcon />}
          className="addBtn"
          onClick={handleAddControlnet}
        >
          more
        </Button>
      </div>
      <div className="specs_container">
        {controlnetArr.map((item, index) => (
          <div className="item" key={item.id}>
            <TextField
              error={item.model_name !== '' ? false : true}
              style={{ minWidth: '60%', marginTop: '10px' }}
              label="Model Name"
              size="small"
              value={item.model_name}
              onChange={(e) => {
                handleUpdateSpecsArr(index, 'model_name', e.target.value)
              }}
            />
            <Box padding="15px"></Box>

            <TextField
              error={item.model_uri !== '' ? false : true}
              style={{ minWidth: '60%' }}
              label="Model Path"
              size="small"
              value={item.model_uri}
              onChange={(e) => {
                handleUpdateSpecsArr(index, 'model_uri', e.target.value)
              }}
            />
            <Box padding="15px"></Box>

            <label
              style={{
                paddingLeft: 5,
              }}
            >
              Model Format
            </label>
            <RadioGroup
              value={item.model_format}
              onChange={(e) => {
                handleUpdateSpecsArr(index, 'model_format', e.target.value)
              }}
            >
              <Box sx={styles.checkboxWrapper}>
                <Box key={item} sx={{ marginLeft: '10px' }}>
                  <FormControlLabel
                    value="controlnet"
                    checked
                    control={<Radio />}
                    label="controlnet"
                  />
                </Box>
              </Box>
            </RadioGroup>

            <Tooltip title="Delete specs" placement="top">
              <div
                className="deleteBtn"
                onClick={() => handleDeleteControlnet(index)}
              >
                <DeleteIcon className="deleteIcon" />
              </div>
            </Tooltip>
          </div>
        ))}
      </div>
    </>
  )
}

export default AddControlnet

const styles = {
  baseFormControl: {
    width: '100%',
    margin: 'normal',
    size: 'small',
  },
  checkboxWrapper: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    width: '100%',
  },
}
