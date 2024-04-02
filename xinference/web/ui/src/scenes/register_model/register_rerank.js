import {
  Box,
  Checkbox,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
} from '@mui/material'
import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Button from '@mui/material/Button'
import TextField from '@mui/material/TextField'
import React, { useContext, useState } from 'react'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import { useMode } from '../../theme'

const SUPPORTED_LANGUAGES_DICT = { en: 'English', zh: 'Chinese' }
// Convert dictionary of supported languages into list
const SUPPORTED_LANGUAGES = Object.keys(SUPPORTED_LANGUAGES_DICT)
const SUPPORTED_HUBS_DICT = { huggingface: 'HuggingFace', modelscope: 'ModelScope' }
const SUPPORTED_HUBS = Object.keys(SUPPORTED_HUBS_DICT)

const SOURCES_DICT = { self_hosted: 'Self Hosted', hub: 'Hub' }
const SOURCES = Object.keys(SOURCES_DICT)

const RegisterRerankModel = () => {
  const ERROR_COLOR = useMode()
  const endPoint = useContext(ApiContext).endPoint
  const { setErrorMsg } = useContext(ApiContext)
  const [successMsg, setSuccessMsg] = useState('')
  const [modelSource, setModelSource] = useState(SOURCES[0])
  const [hub, setHub] = useState(SUPPORTED_HUBS[0])
  const [modelId, setModelId] = useState('')

  const [formData, setFormData] = useState({
    model_name: 'custom-rerank',
    language: ['en'],
    model_uri: '/path/to/rerank-model',
    model_id: null,
    model_hub: null,
  })

  const errorModelName = formData.model_name.trim().length <= 0
  const errorModelId = modelSource === 'hub' && modelId.search('\\w+/\\w+') === -1
  const errorLanguage =
    formData.language === undefined || formData.language.length === 0

  const handleClick = async () => {
    const errorAny = errorModelName || errorLanguage || errorModelId

    if (errorAny) {
      setErrorMsg('Please fill in valid value for all fields')
      return
    }

    try {
      let myFormData
      if (modelSource === 'hub') {
        myFormData = {
          ...formData,
          model_id: modelId,
          model_hub: hub,
          model_uri: null,
        }
      } else {
        myFormData = {
          ...formData,
          model_id: null,
          model_hub: null,
        }
      }
      const response = await fetcher(
        endPoint + '/v1/model_registrations/rerank',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: JSON.stringify(myFormData),
            persist: true,
          }),
        },
      )
      if (!response.ok) {
        const errorData = await response.json() // Assuming the server returns error details in JSON format
        setErrorMsg(
          `Server error: ${response.status} - ${
            errorData.detail || 'Unknown error'
          }`,
        )
      } else {
        setSuccessMsg(
          'Model has been registered successfully! Navigate to launch model page to proceed.',
        )
      }
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error)
      setErrorMsg(error.message || 'An unexpected error occurred.')
    }
  }

  const toggleLanguage = (lang) => {
    if (formData.language.includes(lang)) {
      setFormData({
        ...formData,
        language: formData.language.filter((l) => l !== lang),
      })
    } else {
      setFormData({
        ...formData,
        language: [...formData.language, lang],
      })
    }
  }

  const handleImportModel = async () => {
    if (errorModelId) {
      setErrorMsg('Please fill in valid value for Model Id')
      return
    }

    const split = modelId.split('/')
    if (split.length !== 2) {
      setErrorMsg('Please fill in valid value for Model Id')
      return
    }

    const repo_name = split[1]
    const repo_split = repo_name.split(/[-_]/)
    let lang = 'en'
    for (const seg of repo_split) {
      if (['zh', 'cn', 'chinese'].includes(seg.toLowerCase())) {
        lang = 'zh'
        break
      }
    }
    setFormData({
      ...formData,
      language: [lang],
    })
  }

  return (
    <React.Fragment>
      <Box padding="20px"></Box>
      {/* Base Information */}
      <FormControl sx={styles.baseFormControl}>
        <TextField
          label="Model Name"
          error={errorModelName}
          defaultValue={formData.model_name}
          size="small"
          helperText="Alphanumeric characters with properly placed hyphens and underscores. Must not match any built-in model names."
          onChange={(event) =>
            setFormData({ ...formData, model_name: event.target.value })
          }
        />
        <Box padding="15px"></Box>


        <label
          style={{
            paddingLeft: 5,
          }}
        >
          Model Source
        </label>

        <RadioGroup
          value={modelSource}
          onChange={(e) => {
            setModelSource(e.target.value)
          }}
        >
          <Box sx={styles.checkboxWrapper}>
            {SOURCES.map((item) => (
              <Box sx={{ marginLeft: '10px' }}>
                <FormControlLabel
                  value={item}
                  control={<Radio />}
                  label={SOURCES_DICT[item]}
                />
              </Box>
            ))}
          </Box>
        </RadioGroup>
        <Box padding="15px"></Box>


        {modelSource === 'self_hosted' &&
          <TextField
            label="Model Path"
            size="small"
            value={formData.model_uri}
            onChange={(e) => {
              setFormData({
                ...formData,
                model_uri: e.target.value,
              })
            }}
            helperText="Provide the model directory path."
          />}
        {modelSource === 'hub' &&
          <Box sx={styles.checkboxWrapper}>

            <TextField
              sx={{ width: '400px' }}
              label="Model Id"
              size="small"
              error={errorModelId}
              value={modelId}
              onChange={(e) => {
                setModelId(e.target.value)
              }}
              placeholder="user/repo"
            />

            <FormControl variant="standard"
                         sx={{ marginLeft: '10px' }}>
              <InputLabel id="hub-label">Hub</InputLabel>
              <Select
                labelId="hub-label"
                value={hub}
                label="Hub"
                onChange={(e) => {
                  setHub(e.target.value)
                }}
              >
                {SUPPORTED_HUBS.map((item) => (
                  <MenuItem value={item}>{SUPPORTED_HUBS_DICT[item]}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button
              sx={{ marginLeft: '10px' }}
              variant="contained"
              color="primary"
              onClick={handleImportModel}
            >
              Import Model
            </Button>
          </Box>
        }
        <Box padding="15px"></Box>

        <label
          style={{
            paddingLeft: 5,
            color: errorLanguage ? ERROR_COLOR : 'inherit',
          }}
        >
          Model Languages
        </label>
        <Box sx={styles.checkboxWrapper}>
          {SUPPORTED_LANGUAGES.map((lang) => (
            <Box key={lang} sx={{ marginRight: '10px' }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={formData.language.includes(lang)}
                    onChange={() => toggleLanguage(lang)}
                    name={lang}
                    sx={
                      errorLanguage
                        ? {
                          'color': ERROR_COLOR,
                          '&.Mui-checked': {
                            color: ERROR_COLOR,
                          },
                        }
                        : {}
                    }
                  />
                }
                label={SUPPORTED_LANGUAGES_DICT[lang]}
                style={{
                  paddingLeft: 10,
                  color: errorLanguage ? ERROR_COLOR : 'inherit',
                }}
              />
            </Box>
          ))}
        </Box>
        <Box padding="15px"></Box>
      </FormControl>

      <Box width={'100%'}>
        {successMsg !== '' && (
          <Alert severity="success">
            <AlertTitle>Success</AlertTitle>
            {successMsg}
          </Alert>
        )}
        <Button
          variant="contained"
          color="primary"
          type="submit"
          onClick={handleClick}
        >
          Register Model
        </Button>
      </Box>
    </React.Fragment>
  )
}

export default RegisterRerankModel

const styles = {
  baseFormControl: {
    width: '100%',
    margin: 'normal',
    size: 'small',
  },
  checkboxWrapper: {
    display: 'flex',
    flexWrap: 'wrap',
    maxWidth: '80%',
  },
  labelPaddingLeft: {
    paddingLeft: 5,
  },
  formControlLabelPaddingLeft: {
    paddingLeft: 10,
  },
  buttonBox: {
    width: '100%',
    margin: '20px',
  },
  error: {
    fontWeight: 'bold',
    margin: '5px 0',
    padding: '1px',
    borderRadius: '5px',
  },
}
