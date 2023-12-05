import { TabContext, TabList, TabPanel } from '@mui/lab'
import { Box, Tab } from '@mui/material'
import React from 'react'

import Title from '../../components/Title'
import LaunchEmbedding from './launchEmbedding'
import LaunchLLM from './launchLLM'
import LaunchRerank from './launchRerank'

const LaunchModel = () => {
  const [value, setValue] = React.useState('1')

  const handleTabChange = (event, newValue) => {
    setValue(newValue)
  }

  return (
    <Box m="20px">
      <Title title="Launch Model" />
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList value={value} onChange={handleTabChange} aria-label="tabs">
            <Tab label="Language Models" value="1" />
            <Tab label="Embedding Models" value="2" />
            <Tab label="Rerank Models" value="3" />
          </TabList>
        </Box>
        <TabPanel value="1" sx={{ padding: 0 }}>
          <LaunchLLM />
        </TabPanel>
        <TabPanel value="2" sx={{ padding: 0 }}>
          <LaunchEmbedding />
        </TabPanel>
        <TabPanel value="3" sx={{ padding: 0 }}>
          <LaunchRerank />
        </TabPanel>
      </TabContext>
    </Box>
  )
}

export default LaunchModel
