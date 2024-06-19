import './styles/registerModelStyle.css'

import CheckIcon from '@mui/icons-material/Check'
import FilterNoneIcon from '@mui/icons-material/FilterNone'
import KeyboardDoubleArrowRightIcon from '@mui/icons-material/KeyboardDoubleArrowRight'
import NotesIcon from '@mui/icons-material/Notes'
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Snackbar,
  Stack,
  Switch,
  TextField,
  Tooltip,
} from '@mui/material'
import ClipboardJS from 'clipboard'
import React, { useContext, useEffect, useRef, useState } from 'react'
import { useCookies } from 'react-cookie'
import { useNavigate, useParams } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import fetcher from '../../components/fetcher'
import AddControlnet from './components/addControlnet'
import AddModelSpecs from './components/addModelSpecs'
import languages from './data/languages'
const SUPPORTED_LANGUAGES_DICT = { en: 'English', zh: 'Chinese' }
const SUPPORTED_FEATURES = ['Generate', 'Chat', 'Vision']

// Convert dictionary of supported languages into list
const SUPPORTED_LANGUAGES = Object.keys(SUPPORTED_LANGUAGES_DICT)

const RegisterModelComponent = ({ modelType, customData }) => {
  const endPoint = useContext(ApiContext).endPoint
  const { setErrorMsg } = useContext(ApiContext)
  const [formData, setFormData] = useState(customData)
  const [promptStyles, setPromptStyles] = useState([])
  const [family, setFamily] = useState({
    chat: [],
    generate: [],
  })
  const [isShow, setIsShow] = useState(false)
  const [languagesArr, setLanguagesArr] = useState([])
  const [isContextLengthAlert, setIsContextLengthAlert] = useState(false)
  const [isDimensionsAlert, setIsDimensionsAlert] = useState(false)
  const [isMaxTokensAlert, setIsMaxTokensAlert] = useState(false)
  const [jsonData, setJsonData] = useState('')
  const [isCopySuccess, setIsCopySuccess] = useState(false)
  const [isSpecsArrError, setIsSpecsArrError] = useState(false)
  const scrollRef = useRef(null)
  const [cookie] = useCookies(['token'])
  const navigate = useNavigate()
  const { registerModelType, model_name } = useParams()
  const [specsArr, setSpecsArr] = useState(model_name ? JSON.parse(sessionStorage.getItem('customJsonData')).model_specs : [])
  const [controlnetArr, setControlnetArr] = useState(model_name ? JSON.parse(sessionStorage.getItem('customJsonData')).controlnet : [])

  useEffect(() => {
    if(model_name) {
      const data = JSON.parse(sessionStorage.getItem('customJsonData'))
      
      if (modelType === 'LLM') {
        const lagArr = data.model_lang.filter(item => item !== 'en' && item !== 'zh')
        setLanguagesArr(lagArr)

        const { version, model_name, model_description, context_length, model_lang, model_ability, model_family, model_specs, prompt_style } = data
        const llmData = {
          version,
          model_name,
          model_description,
          context_length,
          model_lang,
          model_ability,
          model_family,
          model_specs,
        }
        prompt_style ? llmData.prompt_style = prompt_style : ''
        setFormData(llmData)
        setSpecsArr(model_specs)
      } else {
        if (modelType === 'embedding') {
          const lagArr = data.language.filter(item => item !== 'en' && item !== 'zh')
          setLanguagesArr(lagArr)

          const { model_name, dimensions, max_tokens, model_uri, language } = data
          const embeddingData = {
            model_name,
            dimensions,
            max_tokens,
            model_uri,
            language,
          }
          setFormData(embeddingData)
        } else if (modelType === 'rerank') {
          const lagArr = data.language.filter(item => item !== 'en' && item !== 'zh')
          setLanguagesArr(lagArr)

          const { model_name, model_uri, language } = data
          const rerankData = {
            model_name,
            model_uri,
            language,
          }
          setFormData(rerankData)
        } else if (modelType === 'image') {
          const { model_name, model_uri, model_family, controlnet } = data
          const imageData = {
            model_name,
            model_uri,
            model_family,
            controlnet,
          }
          setFormData(imageData)
          setControlnetArr(controlnet)
        } else if (modelType === 'audio') {
          const { model_name, model_uri, multilingual, model_family } = data
          const audioData = {
            model_name,
            model_uri,
            multilingual,
            model_family,
          }
          setFormData(audioData)
        }
      }
    }
  }, [model_name])

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

    if (
      Object.prototype.hasOwnProperty.call(customData, 'model_ability') &&
      Object.prototype.hasOwnProperty.call(customData, 'model_family')
    ) {
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
    }
  }, [cookie.token])

  useEffect(() => {
    setJsonData(JSON.stringify(formData, null, 4))
  }, [formData])

  const getFamilyByAbility = () => {
    if (
      formData.model_ability.includes('chat') ||
      formData.model_ability.includes('vision')
    ) {
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
    console.log('formData', modelType, formData)

    for (let key in formData) {
      const type = Object.prototype.toString.call(formData[key]).slice(8, -1)
      if (
        key !== 'model_description' &&
        ((type === 'Array' &&
          key !== 'controlnet' &&
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
      const response = await fetcher(
        endPoint + `/v1/model_registrations/${modelType}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: JSON.stringify(formData),
            persist: true,
          }),
        }
      )
      if (!response.ok) {
        const errorData = await response.json() // Assuming the server returns error details in JSON format
        setErrorMsg(
          `Server error: ${response.status} - ${
            errorData.detail || 'Unknown error'
          }`
        )
      } else {
        navigate(`/launch_model/custom/${modelType.toLowerCase()}`)
        sessionStorage.setItem('modelType', '/launch_model/custom/llm')
        sessionStorage.setItem(
          'subType',
          `/launch_model/custom/${modelType.toLowerCase()}`
        )
      }
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

    // if(value !== '' && Number(value) > 0) {
    //   console.log(111);
    //   setFormData({ ...formData, [parameterName]: Number(value) })
    // }

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
    if (formData.model_ability.includes(ability)) {
      const obj = JSON.parse(JSON.stringify(formData))
      if (ability === 'chat') {
        delete obj.prompt_style
      }
      setFormData({
        ...obj,
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

  const toggleFamily = (value) => {
    const ps = promptStyles.find((item) => item.name === value)
    if (formData.model_ability.includes('chat') && ps) {
      const prompt_style = {
        style_name: ps.style_name,
        system_prompt: ps.system_prompt,
        roles: ps.roles,
        intra_message_sep: ps.intra_message_sep,
        inter_message_sep: ps.inter_message_sep,
        stop: ps.stop ?? null,
        stop_token_ids: ps.stop_token_ids ?? null,
      }
      setFormData({
        ...formData,
        model_family: value,
        prompt_style,
      })
    } else {
      setFormData({ ...formData, model_family: value })
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

  const handleCopy = () => {
    const clipboard = new ClipboardJS('.copyIcon', {
      text: () => jsonData,
    })

    clipboard.on('success', function (event) {
      event.clearSelection()
      setIsCopySuccess(true)
    })
  }

  const handleCancel = () => {
    const model_name = JSON.parse(sessionStorage.getItem('customJsonData')).model_name
    navigate(`/launch_model/custom/${registerModelType}/${model_name}`)
  }

  const handleModify = () => {
    fetcher(
      endPoint +
        `/v1/model_registrations/${
          registerModelType === 'llm' ? 'LLM' : registerModelType
        }/${model_name}`,
      {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )
      .then(() => {
        handleClick()
      })
      .catch((error) => {
        console.error('Error:', error)
      })
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
        <FormControl sx={styles.baseFormControl}>
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
              <Box sx={styles.checkboxWrapper}>
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
            <FormControl sx={styles.baseFormControl}>
              <label
                style={{
                  paddingLeft: 5,
                  color: 'inherit',
                }}
              >
                Model Family
              </label>
              {modelType === 'LLM' && formData.model_family && (
                <Alert
                  icon={<CheckIcon fontSize="inherit" />}
                  severity="success"
                >
                  Please be careful to select the family name corresponding to
                  the model you want to register. If not found, please choose
                  <i style={{ fontStyle: 'italic', fontWeight: 700 }}> other</i>
                  .
                </Alert>
              )}
              {modelType === 'LLM' && !formData.model_family && (
                <Alert severity="error">
                  Please be careful to select the family name corresponding to
                  the model you want to register. If not found, please choose
                  <i style={{ fontStyle: 'italic', fontWeight: 700 }}> other</i>
                  .
                </Alert>
              )}
              <RadioGroup
                value={formData.model_family}
                onChange={(e) => {
                  toggleFamily(e.target.value)
                }}
              >
                <Box
                  sx={styles.checkboxWrapper}
                  style={{ paddingLeft: '10px' }}
                >
                  {modelType === 'LLM' &&
                    sortStringsByFirstLetter(getFamilyByAbility()).map((v) => (
                      <Box sx={{ width: '20%' }} key={v}>
                        <FormControlLabel
                          value={v}
                          control={<Radio />}
                          label={v}
                        />
                      </Box>
                    ))}
                  {(modelType === 'image' || modelType === 'audio') && (
                    <FormControlLabel
                      value={formData.model_family}
                      checked
                      control={<Radio />}
                      label={formData.model_family}
                    />
                  )}
                </Box>
              </RadioGroup>
              <Box padding="15px"></Box>
            </FormControl>
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
        </FormControl>

        {model_name ? (
          <>
            <Button
              variant="contained"
              color="primary"
              type="submit"
              onClick={handleModify}
            >
              Modify
            </Button>
            <Button
              style={{marginLeft: 30}}
              variant="outlined"
              color="primary"
              type="submit"
              onClick={handleCancel}
            >
              Cancel
            </Button>
          </>
        ) : (<Box width={'100%'}>
          <Button
            variant="contained"
            color="primary"
            type="submit"
            onClick={handleClick}
          >
            Register Model
          </Button>
        </Box>)}
      </div>

      {/* JSON */}
      <div className={isShow ? 'jsonBox' : 'jsonBox hide'}>
        <div className="jsonBox-header">JSON Format</div>
        <Tooltip title="Copy all" placement="top">
          <FilterNoneIcon className="copyIcon" onClick={handleCopy} />
        </Tooltip>
        <textarea readOnly className="textarea" value={jsonData} />
      </div>

      <Snackbar
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        open={isCopySuccess}
        autoHideDuration={1500}
        onClose={() => setIsCopySuccess(false)}
      >
        <Alert severity="success" variant="filled" sx={{ width: '100%' }}>
          Copied to clipboard!
        </Alert>
      </Snackbar>
    </Box>
  )
}

export default RegisterModelComponent

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
