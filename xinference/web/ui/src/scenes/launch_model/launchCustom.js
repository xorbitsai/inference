import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, FormControl, Tab } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../../components/apiContext'
import fetchWrapper from '../../components/fetchWrapper'
import HotkeyFocusTextField from '../../components/hotkeyFocusTextField'
import ModelCard from './modelCard'

const customType = ['llm', 'embedding', 'rerank', 'image', 'audio', 'flexible']

const LaunchCustom = ({ gpuAvailable }) => {
  let endPoint = useContext(ApiContext).endPoint
  const [registrationData, setRegistrationData] = useState([])
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)

  // States used for filtering
  const [searchTerm, setSearchTerm] = useState('')
  const [value, setValue] = useState(sessionStorage.getItem('subType'))
  const { t } = useTranslation()

  const navigate = useNavigate()
  const handleTabChange = (_, newValue) => {
    const type =
      newValue.split('/')[3] === 'llm' ? 'LLM' : newValue.split('/')[3]
    getData(type)
    setValue(newValue)
    navigate(newValue)
    sessionStorage.setItem('subType', newValue)
  }

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value)
  }

  const filter = (registration) => {
    if (!registration || typeof searchTerm !== 'string') return false
    const modelName = registration.model_name
      ? registration.model_name.toLowerCase()
      : ''
    return modelName.includes(searchTerm.toLowerCase())
  }

  useEffect(() => {
    const type = sessionStorage.getItem('subType').split('/')[3]
    getData(type === 'llm' ? 'LLM' : type)
  }, [])

  const getData = async (type) => {
    if (isCallingApi || isUpdatingModel) return
    try {
      setIsCallingApi(true)

      const data = await fetchWrapper.get(`/v1/model_registrations/${type}`)
      const customRegistrations = data.filter((data) => !data.is_builtin)

      const newData = await Promise.all(
        customRegistrations.map(async (registration) => {
          const desc = await fetchWrapper.get(
            `/v1/model_registrations/${type}/${registration.model_name}`
          )

          return {
            ...desc,
            is_builtin: registration.is_builtin,
          }
        })
      )
      setRegistrationData(newData)
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsCallingApi(false)
    }
  }

  const handlecustomDelete = (model_name) => {
    setRegistrationData(
      registrationData.filter((item) => {
        return item.model_name !== model_name
      })
    )
  }

  const style = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    paddingLeft: '2rem',
    paddingBottom: '2rem',
    gridGap: '2rem 0rem',
  }

  return (
    <Box m="20px">
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList value={value} onChange={handleTabChange} aria-label="tabs">
            <Tab
              label={t('model.languageModels')}
              value="/launch_model/custom/llm"
            />
            <Tab
              label={t('model.embeddingModels')}
              value="/launch_model/custom/embedding"
            />
            <Tab
              label={t('model.rerankModels')}
              value="/launch_model/custom/rerank"
            />
            <Tab
              label={t('model.imageModels')}
              value="/launch_model/custom/image"
            />
            <Tab
              label={t('model.audioModels')}
              value="/launch_model/custom/audio"
            />
            <Tab
              label={t('model.flexibleModels')}
              value="/launch_model/custom/flexible"
            />
          </TabList>
        </Box>
        {customType.map((item) => (
          <TabPanel
            key={item}
            value={`/launch_model/custom/${item}`}
            sx={{ padding: 0 }}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr',
                margin: '30px 2rem',
              }}
            >
              <FormControl variant="outlined" margin="normal">
                <HotkeyFocusTextField
                  id="search"
                  type="search"
                  label={t('launchModel.search')}
                  value={searchTerm}
                  onChange={handleSearchChange}
                  size="small"
                  hotkey="Enter"
                  t={t}
                />
              </FormControl>
            </div>
            <div style={style}>
              {registrationData
                .filter((registration) => filter(registration))
                .map((filteredRegistration) => (
                  <ModelCard
                    key={filteredRegistration.model_name}
                    url={endPoint}
                    modelData={filteredRegistration}
                    gpuAvailable={gpuAvailable}
                    is_custom={true}
                    modelType={item === 'llm' ? 'LLM' : item}
                    onHandlecustomDelete={handlecustomDelete}
                  />
                ))}
            </div>
          </TabPanel>
        ))}
      </TabContext>
    </Box>
  )
}

export default LaunchCustom
