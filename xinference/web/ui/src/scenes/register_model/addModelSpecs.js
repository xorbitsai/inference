import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete'
import {
  Alert,
  Box,
  Button,
  FormControlLabel,
  Radio,
  RadioGroup,
  TextField,
  Tooltip,
} from '@mui/material'
import React, { useEffect, useState } from 'react'

import styles from './styles/indexStyle'

const modelFormatArr = [
    {value: 'pytorch', label: 'PyTorch'},
    {value: 'ggmlv3', label: 'GGML'},
    {value: 'ggufv2', label: 'GGUF'},
    {value: 'gptq', label: 'GPTQ'},
    {value: 'awq', label: 'AWQ'},
]

const AddModelSpecs = ({onGetArr}) => {
    const [count, setCount] = useState(1)
    const [specsArr, setSpecsArr] = useState([])
    const [modelSizeAlertId, setModelSizeAlertId] = useState([])
    const [quantizationAlertId, setQuantizationAlertId] = useState([])

    useEffect(() => {
        setSpecsArr([{
          id: 0,
          model_uri: '/path/to/llama-2',
          model_size_in_billions: 7,
          model_format: 'pytorch',
          quantizations: ''
      }])
    }, [])

    useEffect(() => {
        const arr = specsArr.map(item => {
          delete item.id
          parseInt(item.model_size_in_billions) === parseFloat(item.model_size_in_billions) 
          ? item.model_size_in_billions = Number(item.model_size_in_billions) 
          : item.model_size_in_billions = item.model_size_in_billions.split('.').join('_')
    
          item.quantizations = ''
          if(item.model_format === 'pytorch') {
            item.quantizations = 'none'
          } else if (item.quantizations === '' && (item.model_format === 'ggmlv3' || item.model_format === 'ggufv2')) {
            item.quantizations = 'default'
          }
        })
        onGetArr(arr)
    }, [specsArr])

    const handleAddSpecs = () => {
        setCount(count + 1)
        const item = {
          id: count,
          model_uri: '/path/to/llama-2',
          model_size_in_billions: 7,
          model_format: 'pytorch',
          quantizations: ''
        }
        setSpecsArr([...specsArr, item])
        console.log(specsArr);
      }
    
      const handleUpdateSpecsArr = (index, type, newValue) => {
        setSpecsArr(
          specsArr.map((item, subIndex) => {
            if (subIndex === index) {
              return { ...item, [type]: newValue }
            }
            return item
          })
        )
      }
    
      const handleDeleteSpecs = (index) => {
        setSpecsArr(
          specsArr.filter((_, subIndex) => index !== subIndex)
        )
      }
    
      const handleModelSize = (index, value, id) => {
        setModelSizeAlertId(
          modelSizeAlertId.filter((item) => item !== id)
        )
        handleUpdateSpecsArr(index, 'model_size_in_billions', value)
        if(value !== '' && (!Number(value) || Number(value) <= 0)) {
          const modelSizeAlertIdArr = Array.from(new Set([...modelSizeAlertId, id]))
          setModelSizeAlertId(modelSizeAlertIdArr)
        }   
      }
    
      const handleQuantization = (model_format, index, value, id) => {
        setQuantizationAlertId(
          quantizationAlertId.filter((item) => item !== id)
        )
        handleUpdateSpecsArr(index, 'quantizations', value)
        if((model_format === 'gptq' || model_format === 'awq') && value === '') {
          const quantizationAlertIdArr = Array.from(new Set([...quantizationAlertId, id]))
          setQuantizationAlertId(quantizationAlertIdArr)
        }
      }


    return (
        <>
            <label style={{marginBottom: '20px'}}>
                Model Specs
                <Button
                  variant="contained"
                  size="small"
                  endIcon={<AddIcon />}
                  className='addBtn'
                  onClick={handleAddSpecs}
                >add Specs</Button>
            </label>
            {JSON.stringify(specsArr)}
            <div className='specs_container'>
            {
              specsArr.map((item, index) => (
                <div className='item' key={item.id}>
                  <TextField
                    label="Model Path"
                    size="small"
                    value={item.model_uri}
                    onChange={(e) => {
                      handleUpdateSpecsArr(index, 'model_uri', e.target.value)
                    }}
                    helperText="For PyTorch, provide the model directory. For GGML/GGUF, provide the model file path."
                  />
                  <Box padding="15px"></Box>

                  <TextField
                    label="Model Size in Billions"
                    size="small"
                    value={item.model_size_in_billions}
                    onChange={(e) => {
                      handleModelSize(index, e.target.value, item.id)
                    }}
                  />
                  {modelSizeAlertId.includes(item.id) && (
                    <Alert severity="error">
                      Please enter a number greater than 0.
                    </Alert>
                  )}
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
                      if(e.target.value === 'gptq' || e.target.value === 'awq') {
                        const quantizationAlertIdArr = Array.from(new Set([...quantizationAlertId, item.id]))
                        setQuantizationAlertId(quantizationAlertIdArr)
                      }
                    }}
                  >
                    <Box sx={styles.checkboxWrapper}>
                      {modelFormatArr.map(item => (
                        <Box key={item.value} sx={{ marginLeft: '10px' }}>
                          <FormControlLabel
                            value={item.value}
                            control={<Radio />}
                            label={item.label}
                          />
                        </Box>
                      ))}
                    </Box>
                  </RadioGroup>
                  

                  {item.model_format !== 'pytorch' && (<>
                    <Box padding="15px"></Box>
                    <TextField
                      label="Quantization (Optional)"
                      size="small"
                      value={item.quantizations}
                      onChange={(e) => {
                        handleQuantization(item.model_format, index, e.target.value, item.id)
                      }}
                      helperText="For GPTQ/AWQ models, please be careful to fill in the quantization corresponding to the model you want to register."
                    />
                    {(item.model_format !== 'ggmlv3' && item.model_format !== 'ggufv2') && quantizationAlertId.includes(item.id) && item.quantizations == '' && (
                      <Alert severity="error">
                        Quantization cannot be left empty.
                      </Alert>
                    )}
                  </>)}

                  {specsArr.length > 1 && <Tooltip title="Delete specs" placement="top">
                    <div className='deleteBtn' onClick={() => handleDeleteSpecs(index)}>
                      <DeleteIcon className='deleteIcon' />
                    </div>
                  </Tooltip>}
                </div>
              ))
            }
          </div>
        </>
    )
}

export default AddModelSpecs
