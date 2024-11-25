import './styles/registerModelStyle.css'

import Cancel from '@mui/icons-material/Cancel'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight'
import NotesIcon from '@mui/icons-material/Notes'
import OpenInFullIcon from '@mui/icons-material/OpenInFull'
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Checkbox,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Stack,
  Switch,
  TextField,
  Tooltip,
} from '@mui/material'
import nunjucks from 'nunjucks'
import React, { useContext, useEffect, useRef, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate, useParams } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import CopyComponent from '../../components/copyComponent/copyComponent'
import fetchWrapper from '../../components/fetchWrapper'
import { isValidBearerToken } from '../../components/utils'
import AddControlnet from './components/addControlnet'
import AddModelSpecs from './components/addModelSpecs'
import AddStop from './components/addStop'
import languages from './data/languages'
const SUPPORTED_LANGUAGES_DICT = { en: 'English', zh: 'Chinese' }
const SUPPORTED_FEATURES = ['Generate', 'Chat', 'Vision', 'Tools']
const messages = [
  {
    role: 'assistant',
    content: 'This is the message content replied by the assistant previously',
  },
  {
    role: 'user',
    content: 'This is the message content sent by the user currently',
  },
]

// Convert dictionary of supported languages into list
const SUPPORTED_LANGUAGES = Object.keys(SUPPORTED_LANGUAGES_DICT)

const RegisterModelComponent = ({ modelType, customData }) => {
  const endPoint = useContext(ApiContext).endPoint
  const { setErrorMsg } = useContext(ApiContext)
  const [formData, setFormData] = useState(customData)
  const [promptStyles, setPromptStyles] = useState([])
  const [family, setFamily] = useState({})
  const [languagesArr, setLanguagesArr] = useState([])
  const [isContextLengthAlert, setIsContextLengthAlert] = useState(false)
  const [isDimensionsAlert, setIsDimensionsAlert] = useState(false)
  const [isMaxTokensAlert, setIsMaxTokensAlert] = useState(false)
  const [jsonData, setJsonData] = useState('')
  const [isSpecsArrError, setIsSpecsArrError] = useState(false)
  const [isValidLauncherArgsAlert, setIsValidLauncherArgsAlert] =
    useState(false)
  const scrollRef = useRef(null)
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()

  const { registerModelType, model_name } = useParams()
  const [isShow, setIsShow] = useState(model_name ? true : false)
  const [specsArr, setSpecsArr] = useState(
    model_name
      ? JSON.parse(sessionStorage.getItem('customJsonData')).model_specs
      : []
  )
  const [controlnetArr, setControlnetArr] = useState(
    model_name
      ? JSON.parse(sessionStorage.getItem('customJsonData')).controlnet
      : []
  )
  const [contrastObj, setContrastObj] = useState({})
  const [isEqual, setIsEqual] = useState(true)
  const [testRes, setTestRes] = useState('')
  const [isOpenMessages, setIsOpenMessages] = useState(false)
  const [testErrorInfo, setTestErrorInfo] = useState('')
  const [isStopTokenIdsAlert, setIsStopTokenIdsAlert] = useState(false)
  const [familyOptions, setFamilyOptions] = useState([])
  const [isEditableFamily, setIsEditableFamily] = useState(true)

  useEffect(() => {
    if (model_name) {
      const data = JSON.parse(sessionStorage.getItem('customJsonData'))

      if (modelType === 'LLM') {
        const lagArr = data.model_lang.filter(
          (item) => item !== 'en' && item !== 'zh'
        )
        setLanguagesArr(lagArr)

        const {
          version,
          model_name,
          model_description,
          context_length,
          model_lang,
          model_ability,
          model_family,
          model_specs,
          chat_template,
          stop_token_ids,
          stop,
        } = data
        const specsDataArr = model_specs.map((item) => {
          const {
            model_uri,
            model_size_in_billions,
            model_format,
            quantizations,
            model_file_name_template,
          } = item
          return {
            model_uri,
            model_size_in_billions,
            model_format,
            quantizations,
            model_file_name_template,
          }
        })
        const llmData = {
          version,
          model_name,
          model_description,
          context_length,
          model_lang,
          model_ability,
          model_family,
          model_specs: specsDataArr,
          chat_template,
          stop_token_ids,
          stop,
        }
        setFormData(llmData)
        setContrastObj(llmData)
        setSpecsArr(specsDataArr)
      } else {
        if (modelType === 'embedding') {
          const lagArr = data.language.filter(
            (item) => item !== 'en' && item !== 'zh'
          )
          setLanguagesArr(lagArr)

          const { model_name, dimensions, max_tokens, model_uri, language } =
            data
          const embeddingData = {
            model_name,
            dimensions,
            max_tokens,
            model_uri,
            language,
          }
          setFormData(embeddingData)
          setContrastObj(embeddingData)
        } else if (modelType === 'rerank') {
          const lagArr = data.language.filter(
            (item) => item !== 'en' && item !== 'zh'
          )
          setLanguagesArr(lagArr)

          const { model_name, model_uri, language } = data
          const rerankData = {
            model_name,
            model_uri,
            language,
          }
          setFormData(rerankData)
          setContrastObj(rerankData)
        } else if (modelType === 'image') {
          const { model_name, model_uri, model_family, controlnet } = data
          const controlnetArr = controlnet.map((item) => {
            const { model_name, model_uri, model_family } = item
            return {
              model_name,
              model_uri,
              model_family,
            }
          })
          const imageData = {
            model_name,
            model_uri,
            model_family,
            controlnet: controlnetArr,
          }
          setFormData(imageData)
          setContrastObj(imageData)
          setControlnetArr(controlnetArr)
        } else if (modelType === 'audio') {
          const { model_name, model_uri, multilingual, model_family } = data
          const audioData = {
            model_name,
            model_uri,
            multilingual,
            model_family,
          }
          setFormData(audioData)
          setContrastObj(audioData)
        } else if (modelType === 'flexible') {
          const {
            model_name,
            model_uri,
            model_description,
            launcher,
            launcher_args,
          } = data
          const flexibleData = {
            model_name,
            model_uri,
            model_description,
            launcher,
            launcher_args,
          }
          setFormData(flexibleData)
          setContrastObj(flexibleData)
        }
      }
    }
  }, [model_name])

  useEffect(() => {
    if (
      sessionStorage.getItem('auth') === 'true' &&
      !isValidBearerToken(sessionStorage.getItem('token')) &&
      !isValidBearerToken(cookie.token)
    ) {
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
        for (let key in data) {
          data[key] = data[key].sort(function (a, b) {
            let lowerA = a.toLowerCase()
            let lowerB = b.toLowerCase()

            if (lowerA < lowerB) {
              return -1
            }
            if (lowerA > lowerB) {
              return 1
            }
            return 0
          })
        }
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

    if (
      Object.prototype.hasOwnProperty.call(customData, 'model_ability') &&
      Object.prototype.hasOwnProperty.call(customData, 'model_family')
    ) {
      if (promptStyles.length === 0) {
        getBuiltInPromptStyles().catch((error) => {
          setErrorMsg(
            error.message ||
              'An unexpected error occurred when getting builtin prompt styles.'
          )
          console.error('Error: ', error)
        })
      }
      if (family?.chat === undefined) {
        getBuiltinFamilies().catch((error) => {
          setErrorMsg(
            error.message ||
              'An unexpected error occurred when getting builtin prompt styles.'
          )
          console.error('Error: ', error)
        })
      }
    }
  }, [cookie.token])

  useEffect(() => {
    setJsonData(JSON.stringify(formData, customReplacer, 4))
    if (contrastObj.model_name) {
      deepEqual(contrastObj, formData) ? setIsEqual(true) : setIsEqual(false)
    }
    if (family?.chat?.length) handleFamilyOptions(formData.model_ability)
  }, [formData])

  useEffect(() => {
    if (family?.chat?.length) handleFamilyOptions(formData.model_ability)
  }, [family])

  const customReplacer = (key, value) => {
    if (key === 'chat_template' && value) {
      return value.replace(/\\n/g, '\n')
    }
    return value
  }

  const handleClick = async () => {
    for (let key in formData) {
      const type = Object.prototype.toString.call(formData[key]).slice(8, -1)
      if (
        key !== 'model_description' &&
        ((type === 'Array' &&
          key !== 'controlnet' &&
          key !== 'stop_token_ids' &&
          key !== 'stop' &&
          formData[key].length === 0) ||
          (type === 'String' && formData[key] === '') ||
          (type === 'Number' && formData[key] <= 0))
      ) {
        setErrorMsg('Please fill in valid value for all fields')
        return
      }
    }

    if (
      isSpecsArrError ||
      isContextLengthAlert ||
      isDimensionsAlert ||
      isMaxTokensAlert
    ) {
      setErrorMsg('Please fill in valid value for all fields')
      return
    }

    try {
      fetchWrapper
        .post(`/v1/model_registrations/${modelType}`, {
          model: JSON.stringify(formData, customReplacer),
          persist: true,
        })
        .then(() => {
          navigate(`/launch_model/custom/${modelType.toLowerCase()}`)
          sessionStorage.setItem('modelType', '/launch_model/custom/llm')
          sessionStorage.setItem(
            'subType',
            `/launch_model/custom/${modelType.toLowerCase()}`
          )
        })
        .catch((error) => {
          console.error('Error:', error)
          if (error.response.status !== 403 && error.response.status !== 401) {
            setErrorMsg(error.message)
          }
        })
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error)
      setErrorMsg(error.message || 'An unexpected error occurred.')
    }
  }

  const handleNumber = (value, parameterName) => {
    setIsContextLengthAlert(false)
    setIsDimensionsAlert(false)
    setIsMaxTokensAlert(false)
    setFormData({ ...formData, [parameterName]: value })

    if (
      value !== '' &&
      (!Number(value) ||
        Number(value) <= 0 ||
        parseInt(value) !== parseFloat(value))
    ) {
      parameterName === 'context_length' ? setIsContextLengthAlert(true) : ''
      parameterName === 'dimensions' ? setIsDimensionsAlert(true) : ''
      parameterName === 'max_tokens' ? setIsMaxTokensAlert(true) : ''
    } else if (value !== '') {
      setFormData({ ...formData, [parameterName]: Number(value) })
    }
  }

  const toggleLanguage = (lang) => {
    if (modelType === 'LLM') {
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
    } else {
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
  }

  const toggleAbility = (ability) => {
    const obj = JSON.parse(JSON.stringify(formData))
    if (formData.model_ability.includes(ability)) {
      delete obj.chat_template
      delete obj.stop_token_ids
      delete obj.stop
      setFormData({
        ...obj,
        model_ability: formData.model_ability.filter((item) => {
          if (ability === 'chat') {
            return item !== 'chat' && item !== 'vision' && item !== 'tools'
          }
          return item !== ability
        }),
        model_family: '',
      })
    } else {
      let model_ability = []
      if (
        ability === 'chat' ||
        (['vision', 'tools'].includes(ability) &&
          !formData.model_ability.includes('chat'))
      ) {
        if (
          formData.model_family !== '' &&
          family?.chat?.includes(formData.model_family)
        ) {
          const data = promptStyles.filter(
            (item) => item.name === formData.model_family
          )
          obj.chat_template = data[0]?.chat_template || null
          obj.stop_token_ids = data[0]?.stop_token_ids || []
          obj.stop = data[0]?.stop || []
        } else {
          obj.chat_template = ''
          obj.stop_token_ids = []
          obj.stop = []
        }
        ability === 'chat'
          ? (model_ability = [...formData.model_ability, ability])
          : (model_ability = [...formData.model_ability, 'chat', ability])
      } else {
        if (ability === 'vision' && formData.model_ability.includes('tools')) {
          model_ability = [
            ...formData.model_ability.filter((item) => item !== 'tools'),
            'chat',
            ability,
          ]
        } else if (
          ability === 'tools' &&
          formData.model_ability.includes('vision')
        ) {
          model_ability = [
            ...formData.model_ability.filter((item) => item !== 'vision'),
            'chat',
            ability,
          ]
        } else {
          model_ability = [...formData.model_ability, ability]
        }
      }
      delete obj.chat_template
      delete obj.stop_token_ids
      delete obj.stop
      setFormData({
        ...obj,
        model_family: '',
        model_ability: model_ability,
      })
    }
  }

  const handleFamily = (value) => {
    if (formData.model_ability.includes('chat')) {
      if (family?.chat?.includes(value)) {
        const data = promptStyles.filter((item) => {
          return item.name === value
        })
        setFormData({
          ...formData,
          model_family: value,
          chat_template: data[0]?.chat_template || null,
          stop_token_ids: data[0]?.stop_token_ids || [],
          stop: data[0]?.stop || [],
        })
      } else {
        setFormData({
          ...formData,
          model_family: value,
          chat_template: '',
          stop_token_ids: [],
          stop: [],
        })
      }
    } else {
      setFormData({
        ...formData,
        model_family: value,
      })
    }
  }

  const handleSelectLanguages = (value) => {
    const arr = [...languagesArr, value]
    setLanguagesArr(arr)
    if (modelType === 'LLM') {
      setFormData({
        ...formData,
        model_lang: Array.from(new Set([...formData.model_lang, ...arr])),
      })
    } else {
      setFormData({
        ...formData,
        language: Array.from(new Set([...formData.language, ...arr])),
      })
    }
  }

  const handleDeleteLanguages = (item) => {
    const arr = languagesArr.filter((subItem) => subItem !== item)
    setLanguagesArr(arr)
    if (modelType === 'LLM') {
      setFormData({
        ...formData,
        model_lang: formData.model_lang.filter((subItem) => subItem !== item),
      })
    } else {
      setFormData({
        ...formData,
        language: formData.language.filter((subItem) => subItem !== item),
      })
    }
  }

  const getSpecsArr = (arr, isSpecsArrError) => {
    setFormData({ ...formData, model_specs: arr })
    setIsSpecsArrError(isSpecsArrError)
  }

  const getControlnetArr = (arr) => {
    setFormData({ ...formData, controlnet: arr })
  }

  const handleCancel = () => {
    navigate(`/launch_model/custom/${registerModelType}`)
  }

  const handleEdit = () => {
    fetchWrapper
      .delete(
        `/v1/model_registrations/${
          registerModelType === 'llm' ? 'LLM' : registerModelType
        }/${model_name}`
      )
      .then(() => handleClick())
      .catch((error) => {
        console.error('Error:', error)
        if (error.response.status !== 403 && error.response.status !== 401) {
          setErrorMsg(error.message)
        }
      })
  }

  const deepEqual = (obj1, obj2) => {
    if (obj1 === obj2) return true
    if (
      typeof obj1 !== 'object' ||
      typeof obj2 !== 'object' ||
      obj1 == null ||
      obj2 == null
    ) {
      return false
    }

    let keysA = Object.keys(obj1)
    let keysB = Object.keys(obj2)
    if (keysA.length !== keysB.length) return false
    for (let key of keysA) {
      if (!keysB.includes(key) || !deepEqual(obj1[key], obj2[key])) {
        return false
      }
    }
    return true
  }

  const handleTest = () => {
    setTestRes('')
    if (formData.chat_template) {
      try {
        nunjucks.configure({ autoescape: false })
        const test_res = nunjucks.renderString(formData.chat_template, {
          messages: messages,
        })
        if (test_res === '') {
          setTestRes(test_res)
          setTestErrorInfo('error')
        } else {
          setTestRes(test_res)
          setTestErrorInfo('')
        }
      } catch (error) {
        setTestErrorInfo(`${error}`)
      }
    }
  }

  const getStopTokenIds = (value) => {
    if (value.length === 1 && value[0] === '') {
      setFormData({
        ...formData,
        stop_token_ids: [],
      })
    } else {
      setFormData({
        ...formData,
        stop_token_ids: value,
      })
    }
  }

  const getStop = (value) => {
    if (value.length === 1 && value[0] === '') {
      setFormData({
        ...formData,
        stop: [],
      })
    } else {
      setFormData({
        ...formData,
        stop: value,
      })
    }
  }

  const handleFamilyAlert = () => {
    if (
      formData.model_ability?.includes('vision') &&
      !family?.vision?.includes(formData.model_family)
    ) {
      return true
    } else if (
      formData.model_ability?.includes('tools') &&
      !family?.tools?.includes(formData.model_family)
    ) {
      return true
    }
    return false
  }

  const handleChatTemplateAlert = () => {
    if (
      familyOptions?.filter((item) => item.id === formData.model_family)
        .length === 0 &&
      !formData.chat_template
    ) {
      return true
    }
    return false
  }

  const handleFamilyOptions = (model_ability) => {
    if (model_ability.includes('vision')) {
      setIsEditableFamily(false)
      setFamilyOptions(
        family?.vision?.map((item) => {
          return {
            id: item,
            label: item,
          }
        })
      )
    } else if (model_ability.includes('tools')) {
      setIsEditableFamily(false)
      setFamilyOptions(
        family?.tools?.map((item) => {
          return {
            id: item,
            label: item,
          }
        })
      )
    } else if (model_ability.includes('chat')) {
      setIsEditableFamily(true)
      setFamilyOptions(
        family?.chat?.map((item) => {
          return {
            id: item,
            label: item,
          }
        })
      )
    } else if (model_ability.includes('generate')) {
      setIsEditableFamily(true)
      setFamilyOptions(
        family?.generate?.map((item) => {
          return {
            id: item,
            label: item,
          }
        })
      )
    } else {
      setIsEditableFamily(true)
      setFamilyOptions([])
    }
  }

  return (
    <Box style={{ display: 'flex', overFlow: 'hidden', maxWidth: '100%' }}>
      <div className="show-json">
        <p>Show custom json config used by api</p>
        {isShow ? (
          <Tooltip title="Pack up" placement="top">
            <KeyboardDoubleArrowRightIcon
              className="icon arrow"
              onClick={() => setIsShow(!isShow)}
            />
          </Tooltip>
        ) : (
          <Tooltip title="Unfold" placement="top">
            <NotesIcon
              className="icon notes"
              onClick={() => setIsShow(!isShow)}
            />
          </Tooltip>
        )}
      </div>
      <div ref={scrollRef} className={isShow ? 'formBox' : 'formBox broaden'}>
        {/* Base Information */}
        <FormControl style={{ width: '100%' }}>
          {/* name */}
          {customData.model_name && (
            <>
              <TextField
                label="Model Name"
                error={formData.model_name ? false : true}
                value={formData.model_name}
                size="small"
                helperText="Alphanumeric characters with properly placed hyphens and underscores. Must not match any built-in model names."
                onChange={(event) =>
                  setFormData({ ...formData, model_name: event.target.value })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* description */}
          {customData.model_description && (
            <>
              <TextField
                label="Model Description (Optional)"
                value={formData.model_description}
                size="small"
                onChange={(event) =>
                  setFormData({
                    ...formData,
                    model_description: event.target.value,
                  })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* Context Length */}
          {customData.context_length && (
            <>
              <TextField
                error={Number(formData.context_length) > 0 ? false : true}
                label="Context Length"
                value={formData.context_length}
                size="small"
                onChange={(event) => {
                  handleNumber(event.target.value, 'context_length')
                }}
              />
              {isContextLengthAlert && (
                <Alert severity="error">
                  Please enter an integer greater than 0.
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}

          {/* dimensions */}
          {customData.dimensions && (
            <>
              <TextField
                label="Dimensions"
                error={Number(formData.dimensions) > 0 ? false : true}
                value={formData.dimensions}
                size="small"
                onChange={(event) => {
                  handleNumber(event.target.value, 'dimensions')
                }}
              />
              {isDimensionsAlert && (
                <Alert severity="error">
                  Please enter an integer greater than 0.
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}

          {/* Max Tokens */}
          {customData.max_tokens && (
            <>
              <TextField
                label="Max Tokens"
                error={Number(formData.max_tokens) > 0 ? false : true}
                value={formData.max_tokens}
                size="small"
                onChange={(event) => {
                  handleNumber(event.target.value, 'max_tokens')
                }}
              />
              {isMaxTokensAlert && (
                <Alert severity="error">
                  Please enter an integer greater than 0.
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}

          {/* path */}
          {customData.model_uri && (
            <>
              <TextField
                label="Model Path"
                error={formData.model_uri ? false : true}
                value={formData.model_uri}
                size="small"
                helperText="Provide the model directory path."
                onChange={(event) =>
                  setFormData({ ...formData, model_uri: event.target.value })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* model_lang */}
          {(customData.model_lang || customData.language) && (
            <>
              <label
                style={{
                  paddingLeft: 5,
                  color:
                    modelType === 'LLM'
                      ? formData.model_lang.length === 0
                        ? '#d32f2f'
                        : 'inherit'
                      : formData.language.length === 0
                      ? '#d32f2f'
                      : 'inherit',
                }}
              >
                Model Languages
              </label>
              <Box className="checkboxWrapper">
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <Box key={lang} sx={{ marginRight: '10px' }}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={
                            modelType === 'LLM'
                              ? formData.model_lang.includes(lang)
                              : formData.language.includes(lang)
                          }
                          onChange={() => toggleLanguage(lang)}
                          name={lang}
                        />
                      }
                      label={SUPPORTED_LANGUAGES_DICT[lang]}
                      style={{
                        paddingLeft: 10,
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
                    {languages
                      .filter((item) => !languagesArr.includes(item.code))
                      .map((item) => (
                        <MenuItem key={item.code} value={item.code}>
                          {item.code}
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>
              </Box>
              <Stack direction="row" spacing={1} style={{ marginLeft: '10px' }}>
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
            </>
          )}

          {/* multilingual */}
          {'multilingual' in customData && (
            <>
              <label
                style={{
                  paddingLeft: 5,
                }}
              >
                Multilingual
              </label>
              <FormControlLabel
                style={{ marginLeft: 0, width: 50 }}
                control={<Switch checked={formData.multilingual} />}
                onChange={(e) =>
                  setFormData({ ...formData, multilingual: e.target.checked })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* abilities */}
          {customData.model_ability && (
            <>
              <label
                style={{
                  paddingLeft: 5,
                  color:
                    formData.model_ability.length == 0 ? '#d32f2f' : 'inherit',
                }}
              >
                Model Abilities
              </label>
              <Box className="checkboxWrapper">
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
                        />
                      }
                      label={ability}
                      style={{
                        paddingLeft: 10,
                      }}
                    />
                  </Box>
                ))}
              </Box>
              <Box padding="15px"></Box>
            </>
          )}

          {/* family */}
          {(customData.model_family === '' || customData.model_family) && (
            <>
              {modelType === 'LLM' && (
                <>
                  <Autocomplete
                    id="free-solo-demo"
                    freeSolo={isEditableFamily}
                    options={familyOptions || []}
                    size="small"
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        helperText={
                          isEditableFamily
                            ? 'You can choose from the built-in models or input your own.'
                            : 'You can only choose from the built-in models.'
                        }
                        InputProps={{
                          ...params.InputProps,
                          disabled: !isEditableFamily,
                        }}
                        label="Model Family"
                      />
                    )}
                    value={formData.model_family}
                    onChange={(_, newValue) => {
                      handleFamily(newValue?.id)
                    }}
                    onInputChange={(_, newInputValue) => {
                      if (isEditableFamily) {
                        handleFamily(newInputValue)
                      }
                    }}
                  />
                  <Box padding="15px"></Box>
                </>
              )}
              {(modelType === 'image' || modelType === 'audio') && (
                <>
                  <FormControl>
                    <label
                      style={{
                        paddingLeft: 5,
                        color: 'inherit',
                      }}
                    >
                      Model Family
                    </label>
                    <RadioGroup value={formData.model_family}>
                      <Box
                        className="checkboxWrapper"
                        style={{ paddingLeft: '10px' }}
                      >
                        <FormControlLabel
                          value={formData.model_family}
                          checked
                          control={<Radio />}
                          label={formData.model_family}
                        />
                      </Box>
                    </RadioGroup>
                  </FormControl>
                  <Box padding="15px"></Box>
                </>
              )}
            </>
          )}

          {/* chat_template */}
          {formData.model_ability?.includes('chat') && (
            <>
              <div className="chat_template_box">
                <TextField
                  label="Chat Template"
                  error={handleChatTemplateAlert()}
                  value={formData.chat_template || ''}
                  size="small"
                  helperText="Please make sure this chat_template passes the test by clicking the TEST button on the right. Please note that this test may not cover all cases and will only be used for the most basic case."
                  multiline
                  rows={6}
                  onChange={(event) =>
                    setFormData({
                      ...formData,
                      chat_template: event.target.value,
                    })
                  }
                  style={{ flex: 1 }}
                />
                <Button
                  variant="contained"
                  onClick={handleTest}
                  style={{ marginTop: 50 }}
                >
                  test
                </Button>
                <div className="chat_template_test">
                  <div className="chat_template_test_mainBox">
                    <div
                      style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                    >
                      <span style={{ fontSize: 16 }}>messages example</span>
                      <OpenInFullIcon
                        onClick={() => setIsOpenMessages(true)}
                        style={{
                          fontSize: 14,
                          color: '#666',
                          cursor: 'pointer',
                        }}
                      />
                    </div>
                    <div>
                      <span
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 10,
                          fontSize: 16,
                        }}
                      >
                        test result
                        {testErrorInfo ? (
                          <Cancel style={{ color: 'red' }} />
                        ) : testRes ? (
                          <CheckCircleIcon
                            style={{ color: 'rgb(46, 125, 50)' }}
                          />
                        ) : (
                          ''
                        )}
                      </span>
                      <div
                        className="test_res_box"
                        style={{
                          backgroundColor:
                            testErrorInfo === ''
                              ? testRes
                                ? 'rgb(237, 247, 237)'
                                : ''
                              : 'rgb(253, 237, 237)',
                        }}
                      >
                        {testErrorInfo !== ''
                          ? testErrorInfo
                          : testRes
                          ? testRes
                          : 'No test results...'}
                      </div>
                    </div>
                  </div>
                  <div
                    className="chat_template_test_tip"
                    style={{ color: testErrorInfo === '' ? '' : '#d32f2f' }}
                  >
                    Please note that failure to pass test may prevent chats from
                    working properly.
                  </div>
                </div>
              </div>
              <Box padding="15px"></Box>
            </>
          )}

          {/* stop_token_ids */}
          {formData.model_ability?.includes('chat') && (
            <>
              <AddStop
                label="Stop Token Ids"
                arrItemType="number"
                formData={formData.stop_token_ids}
                onGetData={getStopTokenIds}
                onGetError={(value) => {
                  if (value.includes('false')) {
                    setIsStopTokenIdsAlert(true)
                  } else {
                    setIsStopTokenIdsAlert(false)
                  }
                }}
                helperText="int type, used to control the stopping of chat models"
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* stop */}
          {formData.model_ability?.includes('chat') && (
            <>
              <AddStop
                label="Stop"
                arrItemType="string"
                formData={formData.stop}
                onGetData={getStop}
                helperText="string type, used to control the stopping of chat models"
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* specs */}
          {customData.model_specs && (
            <>
              <AddModelSpecs
                isJump={model_name ? true : false}
                formData={customData.model_specs[0]}
                specsDataArr={specsArr}
                onGetArr={getSpecsArr}
                scrollRef={scrollRef}
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* controlnet */}
          {customData.controlnet && (
            <>
              <AddControlnet
                controlnetDataArr={controlnetArr}
                onGetControlnetArr={getControlnetArr}
                scrollRef={scrollRef}
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* launcher */}
          {customData.launcher && (
            <>
              <TextField
                label="Launcher"
                error={formData.launcher ? false : true}
                value={formData.launcher}
                size="small"
                helperText="Provide the model launcher."
                onChange={(event) =>
                  setFormData({ ...formData, launcher: event.target.value })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* launcher_args */}
          {customData.launcher_args && (
            <>
              <TextField
                label="Launcher Arguments (Optional)"
                value={formData.launcher_args}
                size="small"
                helperText="A JSON-formatted dictionary representing the arguments passed to the Launcher."
                onChange={(event) => {
                  try {
                    JSON.parse(event.target.value)
                    setIsValidLauncherArgsAlert(false)
                  } catch {
                    setIsValidLauncherArgsAlert(true)
                  }
                  return setFormData({
                    ...formData,
                    launcher_args: event.target.value,
                  })
                }}
                multiline
                rows={4}
              />
              {isValidLauncherArgsAlert && (
                <Alert severity="error">
                  Please enter the JSON-formatted dictionary.
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}
        </FormControl>

        {model_name ? (
          <>
            <Button
              variant="contained"
              color="primary"
              type="submit"
              onClick={handleEdit}
              disabled={isEqual}
            >
              Edit
            </Button>
            <Button
              style={{ marginLeft: 30 }}
              variant="outlined"
              color="primary"
              type="submit"
              onClick={handleCancel}
            >
              Cancel
            </Button>
          </>
        ) : (
          <Box width={'100%'}>
            <Button
              variant="contained"
              color="primary"
              type="submit"
              onClick={handleClick}
              disabled={
                isContextLengthAlert ||
                isDimensionsAlert ||
                isMaxTokensAlert ||
                formData.model_lang?.length === 0 ||
                formData.language?.length === 0 ||
                formData.model_ability?.length === 0 ||
                (modelType === 'LLM' && !formData.model_family) ||
                isStopTokenIdsAlert ||
                handleFamilyAlert()
              }
            >
              Register Model
            </Button>
          </Box>
        )}
      </div>

      <Dialog
        open={isOpenMessages}
        onClose={() => setIsOpenMessages(false)}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">Messages Example</DialogTitle>
        <DialogContent>
          <textarea
            readOnly
            className="textarea"
            style={{ width: 500, height: 200 }}
            value={JSON.stringify(messages, null, 4)}
          />
        </DialogContent>
        <DialogActions>
          <Button
            variant="contained"
            onClick={() => setIsOpenMessages(false)}
            style={{ marginRight: 15, marginBottom: 15 }}
          >
            OK
          </Button>
        </DialogActions>
      </Dialog>

      {/* JSON */}
      <div className={isShow ? 'jsonBox' : 'jsonBox hide'}>
        <div className="jsonBox-header">
          <div className="jsonBox-title">JSON Format</div>
          <CopyComponent tip={'Copy all'} text={jsonData} />
        </div>
        <textarea readOnly className="textarea" value={jsonData} />
      </div>
    </Box>
  )
}

export default RegisterModelComponent
