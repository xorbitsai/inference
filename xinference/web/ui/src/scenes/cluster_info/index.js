import { Box } from '@mui/material'
import Paper from '@mui/material/Paper'
import Grid from '@mui/material/Unstable_Grid2'
import React, { useContext } from 'react'

import { ApiContext } from '../../components/apiContext'
import TableTitle from '../../components/tableTitle'
import Title from '../../components/Title'
import NodeInfo from './nodeInfo'

const ClusterInfo = () => {
  const endPoint = useContext(ApiContext).endPoint

  return (
    <Box
      sx={{
        height: '100%',
        width: '100%',
        paddingLeft: '20px',
        paddingTop: '20px',
      }}
    >
      <Title title="Cluster Information" />
      <Grid container spacing={3} style={{width: '100%'}}>
        <Grid item xs={12}>
          <Paper
            sx={{
              padding: 2,
              display: 'flex',
              overflow: 'auto',
              flexDirection: 'column',
            }}
          >
            <TableTitle>Supervisor</TableTitle>
            <NodeInfo nodeRole="Supervisor" endpoint={endPoint} />
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper
            sx={{
              padding: 2,
              display: 'flex',
              overflow: 'auto',
              flexDirection: 'column',
            }}
          >
            <TableTitle>Workers</TableTitle>
            <NodeInfo nodeRole="Worker" endpoint={endPoint} />
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper
            sx={{
              padding: 2,
              display: 'flex',
              overflow: 'auto',
              flexDirection: 'column',
            }}
          >
            <TableTitle>Worker Details</TableTitle>
            <NodeInfo nodeRole="Worker-Details" endpoint={endPoint} />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default ClusterInfo
