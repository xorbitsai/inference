import './styles/registerModelStyle.css'

import {
  AutoFixHigh,
  Cancel,
  CheckCircle,
  KeyboardDoubleArrowRight,
  Notes,
  OpenInFull,
} from '@mui/icons-material'
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
import React, { useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useTranslation } from 'react-i18next'
import { useNavigate, useParams } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import CopyComponent from '../../components/copyComponent'
import fetchWrapper from '../../components/fetchWrapper'
import { isValidBearerToken } from '../../components/utils'
import AddControlnet from './components/addControlnet'
import AddModelSpecs from './components/addModelSpecs'
import AddStop from './components/addStop'
import AddVirtualenv from './components/addVirtualenv'
import languages from './data/languages'
const SUPPORTED_LANGUAGES_DICT = { en: 'English', zh: 'Chinese' }
const model_ability_options = [
  {
    type: 'LLM',
    options: [
      'Generate',
      'Chat',
      'Vision',
      'Tools',
      'Reasoning',
      'Audio',
      'Omni',
      'Hybrid',
    ],
  },
  {
    type: 'image',
    options: ['text2image', 'image2image', 'inpainting'],
  },
  {
    type: 'audio',
    options: ['text2audio', 'audio2text'],
  },
]
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
const model_family_options = [
  {
    type: 'image',
    options: ['stable_diffusion'],
  },
  {
    type: 'audio',
    options: [
      'whisper',
      'ChatTTS',
      'CosyVoice',
      'F5-TTS',
      'F5-TTS-MLX',
      'FishAudio',
      'Kokoro',
      'MegaTTS',
      'MeloTTS',
      'funasr',
    ],
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
  const [openAutoFillDialog, setOpenAutoFillDialog] = useState(false)
  const [autoFillModelPath, setAutoFillModelPath] = useState('')
  const [autoFillModelFamily, setAutoFillModelFamily] = useState('')
  const { t } = useTranslation()

  const allFamilies = useMemo(() => {
    return [
      ...new Set(
        Object.values(family || {}).reduce(
          (acc, cur) => acc.concat(cur || []),
          []
        )
      ),
    ]
  }, [family])

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
          reasoning_start_tag = '',
          reasoning_end_tag = '',
        } = data
        const virtualenv = data.virtualenv ?? { packages: [] }
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
          ...(model_ability.includes('reasoning') && {
            reasoning_start_tag,
            reasoning_end_tag,
          }),
          virtualenv,
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

          const {
            version,
            model_name,
            dimensions,
            max_tokens,
            model_uri,
            language,
          } = data
          const model_specs = data.model_specs ?? [
            {
              model_uri: model_uri,
              model_format: 'pytorch',
              quantization: 'none',
            },
          ]
          const virtualenv = data.virtualenv ?? { packages: [] }
          const embeddingData = {
            version,
            model_name,
            dimensions,
            max_tokens,
            language,
            model_specs,
            virtualenv,
          }
          setFormData(embeddingData)
          setContrastObj(embeddingData)
        } else if (modelType === 'rerank') {
          const lagArr = data.language.filter(
            (item) => item !== 'en' && item !== 'zh'
          )
          setLanguagesArr(lagArr)

          const {
            version,
            model_name,
            max_tokens = 512,
            model_uri,
            language,
          } = data
          const model_specs = data.model_specs ?? [
            {
              model_uri: model_uri,
              model_format: 'pytorch',
              quantization: 'none',
            },
          ]
          const virtualenv = data.virtualenv ?? { packages: [] }
          const rerankData = {
            version,
            model_name,
            max_tokens,
            language,
            model_specs,
            virtualenv,
          }
          setFormData(rerankData)
          setContrastObj(rerankData)
        } else if (modelType === 'image') {
          const {
            version,
            model_name,
            model_uri,
            model_family,
            model_ability = [],
            controlnet,
          } = data
          const virtualenv = data.virtualenv ?? { packages: [] }
          const controlnetArr = controlnet.map((item) => {
            const { model_name, model_uri, model_family } = item
            return {
              model_name,
              model_uri,
              model_family,
            }
          })
          const imageData = {
            version,
            model_name,
            model_uri,
            model_family,
            model_ability,
            controlnet: controlnetArr,
            virtualenv,
          }
          setFormData(imageData)
          setContrastObj(imageData)
          setControlnetArr(controlnetArr)
        } else if (modelType === 'audio') {
          const {
            version,
            model_name,
            model_uri,
            multilingual,
            model_ability = [],
            model_family,
          } = data
          const virtualenv = data.virtualenv ?? { packages: [] }
          const audioData = {
            version,
            model_name,
            model_uri,
            multilingual,
            model_ability,
            model_family,
            virtualenv,
          }
          setFormData(audioData)
          setContrastObj(audioData)
        } else if (modelType === 'flexible') {
          const {
            version,
            model_name,
            model_uri,
            model_description,
            launcher,
            launcher_args,
          } = data
          const virtualenv = data.virtualenv ?? { packages: [] }
          const flexibleData = {
            version,
            model_name,
            model_uri,
            model_description,
            launcher,
            launcher_args,
            virtualenv,
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
    const fieldsToDelete = [
      'chat_template',
      'stop_token_ids',
      'stop',
      'reasoning_start_tag',
      'reasoning_end_tag',
      'tool_parser',
    ]
    fieldsToDelete.forEach((key) => delete obj[key])

    const currentAbilities = formData.model_ability
    const isRemoving = currentAbilities.includes(ability)
    const chatRelatedAbilities = [
      'chat',
      'vision',
      'tools',
      'reasoning',
      'audio',
      'omni',
      'hybrid',
    ]
    const isChatRelated = chatRelatedAbilities.includes(ability)
    const mutuallyExclusive = { vision: 'tools', tools: 'vision' }

    if (currentAbilities.includes('chat') && ability !== 'chat') {
      obj.chat_template = ''
      obj.stop_token_ids = []
      obj.stop = []
    }

    if (
      currentAbilities.includes('reasoning') &&
      ability !== 'reasoning' &&
      ability !== 'chat'
    ) {
      obj.reasoning_start_tag = ''
      obj.reasoning_end_tag = ''
    }

    if (
      currentAbilities.includes('tools') &&
      ability !== 'tools' &&
      ability !== 'chat'
    ) {
      delete obj.tool_parser
    }

    if (isRemoving) {
      const updatedAbilities =
        ability === 'chat'
          ? currentAbilities.filter(
              (item) => !chatRelatedAbilities.includes(item)
            )
          : currentAbilities.filter((item) => item !== ability)

      if (ability === 'tools') {
        delete obj.tool_parser
      }

      setFormData({
        ...obj,
        model_family: '',
        model_ability: updatedAbilities,
      })
      return
    }

    let model_ability = [...currentAbilities]

    if (ability === 'reasoning') {
      obj.reasoning_start_tag = ''
      obj.reasoning_end_tag = ''
    }

    if (ability === 'tools') {
      obj.tool_parser = ''
    }

    if (
      ability === 'chat' ||
      (isChatRelated && !currentAbilities.includes('chat'))
    ) {
      obj.chat_template = ''
      obj.stop_token_ids = []
      obj.stop = []

      if (ability !== 'chat' && !model_ability.includes('chat')) {
        model_ability.push('chat')
      }
      model_ability.push(ability)
    } else {
      const conflict = mutuallyExclusive[ability]
      if (conflict && model_ability.includes(conflict)) {
        model_ability = model_ability.filter((item) => item !== conflict)
      }
      model_ability.push(ability)
    }

    setFormData({
      ...obj,
      model_family: '',
      model_ability,
    })
  }

  const toggleImageAbility = (ability) => {
    const currentAbilities = formData.model_ability || []
    const isRemoving = currentAbilities.includes(ability)
    const updated = isRemoving
      ? currentAbilities.filter((item) => item !== ability)
      : [...currentAbilities, ability]

    setFormData({
      ...formData,
      model_ability: updated,
    })
  }

  const handleFamily = (value) => {
    if (formData.model_ability.includes('chat')) {
      if (family?.chat?.includes(value)) {
        const data = promptStyles.find((item) => {
          return item.name === value
        })

        const form_data = {
          ...formData,
          model_family: value,
          chat_template: data?.chat_template || null,
          stop_token_ids: data?.stop_token_ids || [],
          stop: data?.stop || [],
        }

        if (formData.model_ability.includes('tools')) {
          form_data.tool_parser = data?.tool_parser || ''
        }

        if (formData.model_ability.includes('reasoning')) {
          form_data.reasoning_start_tag = data?.reasoning_start_tag || ''
          form_data.reasoning_end_tag = data?.reasoning_end_tag || ''
        }

        setFormData(form_data)
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
      if (modelType === 'image') {
        setFormData({
          ...formData,
          model_family: value,
        })
      } else {
        setFormData({
          ...formData,
          model_family: value,
        })
      }
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
    const abilityMap = {
      reasoning: { editable: false, source: family?.reasoning },
      audio: { editable: false, source: family?.audio },
      omni: { editable: false, source: family?.omni },
      hybrid: { editable: false, source: family?.hybrid },
      vision: { editable: false, source: family?.vision },
      tools: { editable: false, source: family?.tools },
      chat: { editable: true, source: family?.chat },
      generate: { editable: true, source: family?.generate },
    }

    const nonEditableGroup = [
      'reasoning',
      'audio',
      'omni',
      'hybrid',
      'vision',
      'tools',
    ]

    const matchedAbilities = Object.keys(abilityMap).filter((key) =>
      model_ability.includes(key)
    )

    const matchedNonEditable = matchedAbilities.filter((key) =>
      nonEditableGroup.includes(key)
    )

    if (matchedNonEditable.length > 1) {
      const baseSource = abilityMap[matchedNonEditable[0]]?.source || []
      let intersection = new Set(baseSource)

      for (let i = 1; i < matchedNonEditable.length; i++) {
        const currentSource = new Set(
          abilityMap[matchedNonEditable[i]]?.source || []
        )
        intersection = new Set(
          [...intersection].filter((item) => currentSource.has(item))
        )
      }

      setIsEditableFamily(false)
      setFamilyOptions(
        Array.from(intersection).map((item) => ({
          id: item,
          label: item,
        }))
      )
      return
    }

    const matched = matchedAbilities[0]
    if (matched) {
      const { editable, source } = abilityMap[matched]
      setIsEditableFamily(editable)
      setFamilyOptions(
        (source || []).map((item) => ({ id: item, label: item }))
      )
    } else {
      setIsEditableFamily(true)
      setFamilyOptions([])
    }
  }

  const changeVirtualenv = (type, index, value) => {
    if (type === 'add') {
      setFormData((prev) => {
        return {
          ...prev,
          virtualenv: {
            ...prev.virtualenv,
            packages: [...prev.virtualenv.packages, ''],
          },
        }
      })
    } else if (type === 'delete') {
      setFormData((prev) => {
        const newPackages = [...prev.virtualenv.packages]
        newPackages.splice(index, 1)

        return {
          ...prev,
          virtualenv: {
            ...prev.virtualenv,
            packages: newPackages,
          },
        }
      })
    } else if (type === 'change') {
      setFormData((prev) => {
        const newPackages = [...prev.virtualenv.packages]
        newPackages[index] = value

        return {
          ...prev,
          virtualenv: {
            ...prev.virtualenv,
            packages: newPackages,
          },
        }
      })
    }
  }

  const autoFillForm = async () => {
    if (!autoFillModelPath || !autoFillModelFamily) {
      setErrorMsg(
        t(
          !autoFillModelPath
            ? 'registerModel.fillInModelPathFirst'
            : 'registerModel.fillInModelFamilyFirst'
        )
      )
      return
    }
    try {
      const res = await fetchWrapper.post('/v1/models/llm/auto-register', {
        model_path: autoFillModelPath,
        model_family: autoFillModelFamily,
      })
      setFormData((prev) => ({ ...prev, ...res }))
      setContrastObj(res)
      setSpecsArr(res.model_specs)

      setOpenAutoFillDialog(false)
    } catch (error) {
      console.error('Error:', error)
      if (error?.response?.status !== 403 && error?.response?.status !== 401) {
        setErrorMsg(error.message)
      }
    }
  }

  return (
    <Box style={{ display: 'flex', overFlow: 'hidden', maxWidth: '100%' }}>
      <div className="show-json">
        <p>{t('registerModel.showCustomJsonConfig')}</p>
        {isShow ? (
          <Tooltip title={t('registerModel.packUp')} placement="top">
            <KeyboardDoubleArrowRight
              className="icon arrow"
              onClick={() => setIsShow(!isShow)}
            />
          </Tooltip>
        ) : (
          <Tooltip title={t('registerModel.unfold')} placement="top">
            <Notes className="icon notes" onClick={() => setIsShow(!isShow)} />
          </Tooltip>
        )}
      </div>
      <div ref={scrollRef} className={isShow ? 'formBox' : 'formBox broaden'}>
        {modelType === 'LLM' && (
          <>
            <Button
              variant="contained"
              color="primary"
              startIcon={<AutoFixHigh />}
              onClick={() => setOpenAutoFillDialog(true)}
            >
              {t('registerModel.autoFill')}
            </Button>
            <Box padding="15px"></Box>
          </>
        )}

        {/* Base Information */}
        <FormControl style={{ width: '100%' }}>
          {/* name */}
          {customData.model_name && (
            <>
              <TextField
                label={t('registerModel.modelName')}
                error={formData.model_name ? false : true}
                value={formData.model_name}
                size="small"
                helperText={t(
                  'registerModel.alphanumericWithHyphensUnderscores'
                )}
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
                label={t('registerModel.modelDescription')}
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
                label={t('registerModel.contextLength')}
                value={formData.context_length}
                size="small"
                onChange={(event) => {
                  handleNumber(event.target.value, 'context_length')
                }}
              />
              {isContextLengthAlert && (
                <Alert severity="error">
                  {t('registerModel.enterIntegerGreaterThanZero')}
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}

          {/* dimensions */}
          {customData.dimensions && (
            <>
              <TextField
                label={t('registerModel.dimensions')}
                error={Number(formData.dimensions) > 0 ? false : true}
                value={formData.dimensions}
                size="small"
                onChange={(event) => {
                  handleNumber(event.target.value, 'dimensions')
                }}
              />
              {isDimensionsAlert && (
                <Alert severity="error">
                  {t('registerModel.enterIntegerGreaterThanZero')}
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}

          {/* Max Tokens */}
          {customData.max_tokens && (
            <>
              <TextField
                label={t('registerModel.maxTokens')}
                error={Number(formData.max_tokens) > 0 ? false : true}
                value={formData.max_tokens}
                size="small"
                onChange={(event) => {
                  handleNumber(event.target.value, 'max_tokens')
                }}
              />
              {isMaxTokensAlert && (
                <Alert severity="error">
                  {t('registerModel.enterIntegerGreaterThanZero')}
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}

          {/* path */}
          {customData.model_uri && (
            <>
              <TextField
                label={t('registerModel.modelPath')}
                error={formData.model_uri ? false : true}
                value={formData.model_uri}
                size="small"
                helperText={t('registerModel.provideModelDirectoryPath')}
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
                {t('registerModel.modelLanguages')}
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
                  <InputLabel>{t('registerModel.languages')}</InputLabel>
                  <Select
                    value={''}
                    label={t('registerModel.languages')}
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
                {t('registerModel.multilingual')}
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
                {t('registerModel.modelAbilities')}
              </label>
              <Box className="checkboxWrapper">
                {modelType === 'LLM' ? (
                  <>
                    {model_ability_options
                      .find((item) => item.type === modelType)
                      .options.map((ability) => (
                        <Box key={ability} sx={{ marginRight: '10px' }}>
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={formData.model_ability.includes(
                                  ability.toLowerCase()
                                )}
                                onChange={() =>
                                  toggleAbility(ability.toLowerCase())
                                }
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
                  </>
                ) : modelType === 'image' ? (
                  <Box className="checkboxWrapper">
                    {model_ability_options
                      .find((item) => item.type === modelType)
                      .options.map((ability) => {
                        return (
                          <Box key={ability} sx={{ marginRight: '10px' }}>
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={formData.model_ability.includes(
                                    ability
                                  )}
                                  onChange={() => toggleImageAbility(ability)}
                                  name={ability}
                                />
                              }
                              label={ability}
                              style={{
                                paddingLeft: 10,
                              }}
                            />
                          </Box>
                        )
                      })}
                  </Box>
                ) : (
                  <RadioGroup
                    value={formData.model_ability}
                    style={{ display: 'flex' }}
                    onChange={(e) => {
                      setFormData({
                        ...formData,
                        model_ability: [e.target.value],
                      })
                    }}
                  >
                    <Box
                      className="checkboxWrapper"
                      sx={{ paddingLeft: '10px' }}
                    >
                      {model_ability_options
                        .find((item) => item.type === modelType)
                        .options.map((subItem) => (
                          <FormControlLabel
                            key={subItem}
                            value={subItem}
                            control={<Radio />}
                            label={subItem}
                          />
                        ))}
                    </Box>
                  </RadioGroup>
                )}
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
                            ? t('registerModel.chooseBuiltInOrCustomModel')
                            : t('registerModel.chooseOnlyBuiltInModel')
                        }
                        InputProps={{
                          ...params.InputProps,
                          disabled: !isEditableFamily,
                        }}
                        label={t('registerModel.modelFamily')}
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
                      {t('registerModel.modelFamily')}
                    </label>
                    <RadioGroup
                      value={formData.model_family}
                      style={{ display: 'flex' }}
                      onChange={(e) => {
                        setFormData({
                          ...formData,
                          model_family: e.target.value,
                        })
                      }}
                    >
                      <Box
                        className="checkboxWrapper"
                        sx={{ paddingLeft: '10px' }}
                      >
                        {model_family_options
                          .find((item) => item.type === modelType)
                          .options.map((subItem) => (
                            <FormControlLabel
                              key={subItem}
                              value={subItem}
                              control={<Radio />}
                              label={subItem}
                            />
                          ))}
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
                  label={t('registerModel.chatTemplate')}
                  error={handleChatTemplateAlert()}
                  value={formData.chat_template || ''}
                  size="small"
                  helperText={t('registerModel.ensureChatTemplatePassesTest')}
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
                  {t('registerModel.test')}
                </Button>
                <div className="chat_template_test">
                  <div className="chat_template_test_mainBox">
                    <div
                      style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                    >
                      <span style={{ fontSize: 16 }}>
                        {t('registerModel.messagesExample')}
                      </span>
                      <OpenInFull
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
                        {t('registerModel.testResult')}
                        {testErrorInfo ? (
                          <Cancel style={{ color: 'red' }} />
                        ) : testRes ? (
                          <CheckCircle style={{ color: 'rgb(46, 125, 50)' }} />
                        ) : (
                          ''
                        )}
                      </span>
                      <div
                        className="test_res_box"
                        style={{
                          borderColor:
                            testErrorInfo === ''
                              ? testRes
                                ? 'rgb(46, 125, 50)'
                                : '#ddd'
                              : 'rgb(46, 125, 50)',
                        }}
                      >
                        {testErrorInfo !== ''
                          ? testErrorInfo
                          : testRes
                          ? testRes
                          : t('registerModel.noTestResults')}
                      </div>
                    </div>
                  </div>
                  <div
                    className="chat_template_test_tip"
                    style={{ color: testErrorInfo === '' ? '' : '#d32f2f' }}
                  >
                    {t('registerModel.testFailurePreventsChatWorking')}
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
                label={t('registerModel.stopTokenIds')}
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
                helperText={t('registerModel.stopControlForChatModels')}
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* stop */}
          {formData.model_ability?.includes('chat') && (
            <>
              <AddStop
                label={t('registerModel.stop')}
                arrItemType="string"
                formData={formData.stop}
                onGetData={getStop}
                helperText={t('registerModel.stopControlStringForChatModels')}
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* reasoning_start_tag */}
          {formData.model_ability?.includes('reasoning') && (
            <>
              <TextField
                label={t('registerModel.reasoningStartTag')}
                value={formData.reasoning_start_tag}
                size="small"
                onChange={(event) =>
                  setFormData({
                    ...formData,
                    reasoning_start_tag: event.target.value,
                  })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* reasoning_end_tag */}
          {formData.model_ability?.includes('reasoning') && (
            <>
              <TextField
                label={t('registerModel.reasoningEndTag')}
                value={formData.reasoning_end_tag}
                size="small"
                onChange={(event) =>
                  setFormData({
                    ...formData,
                    reasoning_end_tag: event.target.value,
                  })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* tool_parser */}
          {formData.model_ability?.includes('tools') && (
            <>
              <TextField
                label={t('registerModel.toolParser')}
                value={formData.tool_parser}
                size="small"
                onChange={(event) =>
                  setFormData({
                    ...formData,
                    tool_parser: event.target.value,
                  })
                }
              />
              <Box padding="15px"></Box>
            </>
          )}

          {/* specs */}
          {customData.model_specs && (
            <>
              <AddModelSpecs
                isJump={
                  (model_name ? true : false) ||
                  (specsArr && specsArr.length > 0)
                }
                formData={
                  (formData?.model_specs && formData.model_specs[0]) ||
                  customData.model_specs[0]
                }
                specsDataArr={specsArr}
                onGetArr={getSpecsArr}
                scrollRef={scrollRef}
                modelType={modelType}
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
                label={t('registerModel.launcher')}
                error={formData.launcher ? false : true}
                value={formData.launcher}
                size="small"
                helperText={t('registerModel.provideModelLauncher')}
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
                label={t('registerModel.launcherArguments')}
                value={formData.launcher_args}
                size="small"
                helperText={t('registerModel.jsonArgumentsForLauncher')}
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
                  {t('registerModel.enterJsonFormattedDictionary')}
                </Alert>
              )}
              <Box padding="15px"></Box>
            </>
          )}

          {/* virtualenv */}
          {customData.virtualenv && (
            <>
              <AddVirtualenv
                virtualenv={formData.virtualenv}
                onChangeVirtualenv={changeVirtualenv}
                scrollRef={scrollRef}
              />
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
              {t('registerModel.edit')}
            </Button>
            <Button
              style={{ marginLeft: 30 }}
              variant="outlined"
              color="primary"
              type="submit"
              onClick={handleCancel}
            >
              {t('registerModel.cancel')}
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
              {t('registerModel.registerModel')}
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
        <DialogTitle id="alert-dialog-title">
          {t('registerModel.messagesExample')}
        </DialogTitle>
        <DialogContent style={{ width: 600 }}>
          <TextField
            multiline
            fullWidth
            rows={10}
            disabled
            defaultValue={JSON.stringify(messages, null, 4)}
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

      <Dialog
        open={openAutoFillDialog}
        onClose={() => setOpenAutoFillDialog(false)}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          {t('registerModel.autoFill')}
        </DialogTitle>
        <DialogContent style={{ width: 600, padding: 20 }}>
          <TextField
            label={t('registerModel.modelPath')}
            value={autoFillModelPath}
            size="small"
            fullWidth
            onChange={(e) => setAutoFillModelPath(e.target.value)}
          />
          <Box padding="15px"></Box>

          <FormControl fullWidth size="small">
            <InputLabel>{t('registerModel.modelFamily')}</InputLabel>
            <Select
              value={autoFillModelFamily}
              label={t('registerModel.modelFamily')}
              onChange={(e) => setAutoFillModelFamily(e.target.value)}
              MenuProps={{
                PaperProps: {
                  style: {
                    maxHeight: 300,
                  },
                },
              }}
            >
              {(allFamilies || []).map((subItem) => (
                <MenuItem key={subItem} value={subItem}>
                  {subItem}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button
            variant="contained"
            onClick={autoFillForm}
            style={{ marginRight: 15, marginBottom: 15 }}
          >
            {t('registerModel.autoFill')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* JSON */}
      <div className={isShow ? 'jsonBox' : 'jsonBox hide'}>
        <div className="jsonBox-header">
          <div className="jsonBox-title">{t('registerModel.JSONFormat')}</div>
          <CopyComponent tip={t('registerModel.copyAll')} text={jsonData} />
        </div>
        <textarea readOnly className="textarea" value={jsonData} />
      </div>
    </Box>
  )
}

export default RegisterModelComponent
