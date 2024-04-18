import './styles/registerLLMStyle.css'

import FilterNoneIcon from '@mui/icons-material/FilterNone';
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight';
import NotesIcon from '@mui/icons-material/Notes';
import {
  // Alert,
  // AlertTitle,
  Box,
  Button,
  Checkbox,
  Chip,
  FormControl,
  FormControlLabel,
  FormHelperText,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Stack,
  TextField,
  Tooltip,
} from '@mui/material'
import React, { useContext, useEffect, useRef, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
// import fetcher from '../../components/fetcher'
import { useMode } from '../../theme'
import AddModelSpecs from './addModelSpecs'
import languages from './data/languages'

const SUPPORTED_LANGUAGES_DICT = { en: 'English', zh: 'Chinese' }
const SUPPORTED_FEATURES = ['Generate', 'Chat', 'Vision']

// Convert dictionary of supported languages into list
const SUPPORTED_LANGUAGES = Object.keys(SUPPORTED_LANGUAGES_DICT)

const RegisterLLM = () => {
  const ERROR_COLOR = useMode()
  const endPoint = useContext(ApiContext).endPoint
  const { setErrorMsg } = useContext(ApiContext)
  // const [successMsg, setSuccessMsg] = useState('')
  const [formData, setFormData] = useState({
    version: 1,
    context_length: 2048,
    model_name: 'custom-llama-2',
    model_lang: ['en'],
    model_ability: ['generate'],
    // model_description: 'This is a custom model description.',
    model_family: '',
    model_specs: [],
    prompt_style: undefined,
  })
  const [promptStyles, setPromptStyles] = useState([])
  const [family, setFamily] = useState({
    chat: [],
    generate: [],
  })
  const [familyLabel, setFamilyLabel] = useState('')
  const [isShow, setIsShow] = useState(false)
  const [languagesArr, setLanguagesArr] = useState([])
  const [modelSpecsArr, setModelSpecsArr] = useState([])
  const [jsonData, setJsonData] = useState(null)
  const [isTextareaError, setIsTextareaError] = useState(false)

  const scrollRef = useRef(null)

  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()

  const errorModelName = formData.model_name.trim().length <= 0
  // const errorModelDescription = formData.model_description.length < 0
  const errorContextLength = formData.context_length === 0
  const errorLanguage =
    formData.model_lang === undefined || formData.model_lang.length === 0
  const errorAbility =
    formData.model_ability === undefined || formData.model_ability.length === 0
  // const errorModelSize =
  //   formData.model_specs &&
  //   formData.model_specs.some((spec) => {
  //     return (
  //       spec.model_size_in_billions === undefined ||
  //       spec.model_size_in_billions === 0
  //     )
  //   })
  const errorFamily = familyLabel === ''
  const errorAny =
    errorModelName ||
    // errorModelDescription ||
    errorContextLength ||
    errorLanguage ||
    errorAbility ||
    // errorModelSize ||
    errorFamily

  useEffect(() => {
    if (cookie.token === '' || cookie.token === undefined) {
      return
    }
    if (cookie.token !== 'no_auth' && !sessionStorage.getItem('token')) {
      navigate('/login', { replace: true })
      return
    }

    const getBuiltinFamilies = async () => {
      const response = await fetch(endPoint + '/v1/models/families', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      if (!response.ok) {
        const errorData = await response.json() // Assuming the server returns error details in JSON format
        setErrorMsg(
          `Server error: ${response.status} - ${
            errorData.detail || 'Unknown error'
          }`
        )
      } else {
        const data = await response.json()
        data.chat.push('other')
        data.generate.push('other')
        setFamily(data)
      }
    }

    const getBuiltInPromptStyles = async () => {
      const response = await fetch(endPoint + '/v1/models/prompts', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      if (!response.ok) {
        const errorData = await response.json() // Assuming the server returns error details in JSON format
        setErrorMsg(
          `Server error: ${response.status} - ${
            errorData.detail || 'Unknown error'
          }`
        )
      } else {
        const data = await response.json()
        let res = []
        for (const key in data) {
          let v = data[key]
          v['name'] = key
          res.push(v)
        }
        setPromptStyles(res)
      }
    }
    // avoid keep requesting backend to get prompts
    if (promptStyles.length === 0) {
      getBuiltInPromptStyles().catch((error) => {
        setErrorMsg(
          error.message ||
            'An unexpected error occurred when getting builtin prompt styles.'
        )
        console.error('Error: ', error)
      })
    }
    if (family.chat.length === 0) {
      getBuiltinFamilies().catch((error) => {
        setErrorMsg(
          error.message ||
            'An unexpected error occurred when getting builtin prompt styles.'
        )
        console.error('Error: ', error)
      })
    }
  }, [cookie.token])

  useEffect(() => {
    setJsonData(JSON.stringify(formData))
  }, [formData])

  const getFamilyByAbility = () => {
    if (formData.model_ability.includes('chat') || formData.model_ability.includes('vision')) {
      return family.chat
    } else {
      return family.generate
    }
  }
  const sortStringsByFirstLetter = (arr) => {
    return arr.sort((a, b) => {
      const firstCharA = a.charAt(0).toLowerCase()
      const firstCharB = b.charAt(0).toLowerCase()
      if (firstCharA < firstCharB) {
        return -1
      }
      if (firstCharA > firstCharB) {
        return 1
      }
      return 0
   })
  }

  const handleClick = async () => {
    formData.model_lang = Array.from(new Set([...formData.model_lang, ...languagesArr]))
    formData.model_family = familyLabel
    formData.model_specs = modelSpecsArr

    if (formData.model_ability.includes('chat')) {
      const ps = promptStyles.find((item) => item.name === familyLabel)
      if (ps) {
        formData.prompt_style = {
          style_name: ps.style_name,
          system_prompt: ps.system_prompt,
          roles: ps.roles,
          intra_message_sep: ps.intra_message_sep,
          inter_message_sep: ps.inter_message_sep,
          stop: ps.stop ?? null,
          stop_token_ids: ps.stop_token_ids ?? null,
        }
      }
    }

    if (errorAny) {
      setErrorMsg('Please fill in valid value for all fields')
      return
    }
    console.log('formData', formData);

    // try {
    //   const response = await fetcher(endPoint + '/v1/model_registrations/LLM', {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({
    //       model: JSON.stringify(formData),
    //       persist: true,
    //     }),
    //   })
    //   if (!response.ok) {
    //     const errorData = await response.json() // Assuming the server returns error details in JSON format
    //     setErrorMsg(
    //       `Server error: ${response.status} - ${
    //         errorData.detail || 'Unknown error'
    //       }`
    //     )
    //   } else {
    //     setSuccessMsg(
    //       'Model has been registered successfully! Navigate to launch model page to proceed.'
    //     )
    //     navigate('/launch_model/custom/llm')
    //     sessionStorage.setItem('modelType', '/launch_model/custom/llm')
    //     sessionStorage.setItem('subType', '/launch_model/custom/llm')
    //   }
    // } catch (error) {
    //   console.error('There was a problem with the fetch operation:', error)
    //   setErrorMsg(error.message || 'An unexpected error occurred.')
    // }
  }

  const toggleLanguage = (lang) => {
    if (formData.model_lang.includes(lang)) {
      setFormData({
        ...formData,
        model_lang: formData.model_lang.filter((l) => l !== lang),
      })
    } else {
      setFormData({
        ...formData,
        model_lang: [...formData.model_lang, lang],
      })
    }
  }

  const toggleAbility = (ability) => {
    if (formData.model_ability.includes(ability)) {
      setFormData({
        ...formData,
        model_ability: formData.model_ability.filter((a) => a !== ability),
        model_family: '',
      })
    } else {
      setFormData({
        ...formData,
        model_ability: [...formData.model_ability, ability],
        model_family: '',
      })
    }
  }

  const handleSelectLanguages = (value) => {
    const arr = [...languagesArr, value]
    setLanguagesArr(arr)
    setFormData({...formData, model_lang: [...formData.model_lang, ...arr]})
  }

  const handleDeleteLanguages = (item) => {
    const arr = languagesArr.filter((subItem) => subItem !== item)
    setLanguagesArr(arr)
    setFormData({...formData, model_lang: formData.model_lang.filter((subItem) => subItem !== item)})
  }

  const getSpecsArr = (arr) => {
    setModelSpecsArr(arr)
    setFormData({...formData, model_specs: arr})
  }

  const handleChangeText = (value) => {
    try {
      JSON.parse(value)
      // setJsonData()
    } catch (error) {
      setIsTextareaError(true)
    }
    setJsonData(value)

    console.log('text-value', value, jsonData);
  }

  const handleCopy = () => {
    console.log('复制');
  }

  // const handleScrollBottom = () => {
  //   scrollRef.current.scrollTo({
  //     top: scrollRef.current.scrollHeight,
  //     behavior: 'smooth',
  //   })
  // }

  return (
    <Box style={{display: 'flex'}}>
      <div ref={scrollRef} className={isShow ? 'formBox' : ' show formBox' }>
        {isShow ? 
        <Tooltip title="Pack up" placement="top"><KeyboardDoubleArrowRightIcon className='icon arrow' onClick={() => setIsShow(!isShow)} /></Tooltip>
        : <Tooltip title="Unfold" placement="top"><NotesIcon className='icon notes' onClick={() => setIsShow(!isShow)} /></Tooltip> }

        {/* Base Information */}
        <FormControl sx={styles.baseFormControl}>
          {/* name */}
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

          {/* description */}
          <TextField
            label="Model Description (Optional)"
            // error={errorModelDescription}
            defaultValue={formData.model_description}
            size="small"
            onChange={(event) =>
              setFormData({
                ...formData,
                model_description: event.target.value,
              })
            }
          />
          <Box padding="15px"></Box>

          {/* Context Length */}
          <TextField
            error={errorContextLength}
            label="Context Length"
            value={formData.context_length}
            size="small"
            onChange={(event) => {
              let value = event.target.value
              // Remove leading zeros
              if (/^0+/.test(value)) {
                value = value.replace(/^0+/, '') || '0'
              }
              // Ensure it's a positive integer, if not set it to the minimum
              if (!/^\d+$/.test(value) || parseInt(value) < 0) {
                value = '0'
              }
              // Update with the processed value
              setFormData({
                ...formData,
                context_length: Number(value),
              })
            }}
          />
          <Box padding="15px"></Box>

          {/* languages */}
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
                      checked={formData.model_lang.includes(lang)}
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
            <FormControl sx={{ m: 1, minWidth: 120 }} size="small">
              <InputLabel>Languages</InputLabel>
              <Select
                value={''}
                label="Languages"
                onChange={(e) => handleSelectLanguages(e.target.value)}
                MenuProps={{
                  PaperProps: {
                    style: { maxHeight: '20vh' },
                  },
                }}
              >
                {
                  languages.filter(item => !languagesArr.includes(item.code)).map((item) => (
                    <MenuItem key={item.code} value={item.code}>{item.code}</MenuItem>
                  ))
                }
              </Select>
            </FormControl>     
          </Box>
          <Stack direction="row" spacing={1} style={{marginLeft: '10px'}}>
            {languagesArr.map((item) => (
              <Chip
                key={item}
                label={item}
                variant="outlined"
                size="small"
                color="primary"
                onDelete={() => handleDeleteLanguages(item)}
              />
            ))}
          </Stack>
          <Box padding="15px"></Box>

          {/* abilities */}
          <label
            style={{
              paddingLeft: 5,
              color: errorAbility ? ERROR_COLOR : 'inherit',
            }}
          >
            Model Abilities
          </label>
          <Box sx={styles.checkboxWrapper}>
            {SUPPORTED_FEATURES.map((ability) => (
              <Box key={ability} sx={{ marginRight: '10px' }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.model_ability.includes(
                        ability.toLowerCase()
                      )}
                      onChange={() => toggleAbility(ability.toLowerCase())}
                      name={ability}
                      sx={
                        errorAbility
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
                  label={ability}
                  style={{
                    paddingLeft: 10,
                    color: errorAbility ? ERROR_COLOR : 'inherit',
                  }}
                />
              </Box>
            ))}
          </Box>
          <Box padding="15px"></Box>

          {/* family */}
          <FormControl sx={styles.baseFormControl}>
            <label
              style={{
                paddingLeft: 5,
                color: errorAbility ? ERROR_COLOR : 'inherit',
              }}
            >
              Model Family
            </label>
            <FormHelperText>
              Please be careful to select the family name corresponding to the
              model you want to register. If not found, please choose `other`.
            </FormHelperText>
            <RadioGroup
              value={formData.model_family}
              onChange={(e) => {
                setFamilyLabel(e.target.value)
                
                setFormData({...formData, model_family: e.target.value})
              }}
            >
              <Box sx={styles.checkboxWrapper} style={{marginLeft: '10px' }}>
                {sortStringsByFirstLetter(getFamilyByAbility()).map((v) => (
                  <Box sx={{ width: '20%' }} key={v}>
                    <FormControlLabel value={v} control={<Radio />} label={v} />
                  </Box>
                ))}
              </Box>
            </RadioGroup>
            <Box padding="15px"></Box>
          </FormControl>

          {/* specs */}
          {/* ----------------------------------------------------------------- */}
          {/* {JSON.stringify(specsArr)} */}
          <AddModelSpecs onGetArr={getSpecsArr} />

          <Box padding="15px"></Box>
        </FormControl>

        <Box width={'100%'}>
          {/* {successMsg !== '' && (
            <Alert severity="success">
              <AlertTitle>Success</AlertTitle>
              {successMsg}
            </Alert>
          )} */}
          <Button
            variant="contained"
            color="primary"
            type="submit"
            onClick={handleClick}
          >
            Register Model
          </Button>
        </Box>
      </div>

      {/* JSON */}
      <div className={isShow ? 'jsonBox' : 'jsonBox hide'}>
        <div className='jsonBox-header'>
          JSON Format
          {/* {isTextareaError && <p>JSON格式错误</p> } */}
        </div>
        <Tooltip title="Copy all" placement="top">
          <FilterNoneIcon className='icon copyIcon' onClick={handleCopy} />
        </Tooltip>
        <textarea 
          className={isTextareaError ? 'textarea textareaError' : 'textarea'}
          placeholder="在此输入......"
          onChange={(e) => handleChangeText(e.target.value)}
          value={JSON.stringify(formData, null, '\t')}
        >
          
        </textarea>
      </div>
    </Box>
  )
}

export default RegisterLLM

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
    width: '100%'
  },
}
