import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Grid from '@mui/material/Unstable_Grid2'
import PropTypes from 'prop-types'
import React from 'react'

import fetchWrapper from '../../components/fetchWrapper'
import { toReadableSize } from '../../components/utils'
import { StyledTableCell, StyledTableRow } from './style'

class NodeInfo extends React.Component {
  constructor(props) {
    super(props)
    this.nodeRole = props.nodeRole
    this.endpoint = props.endpoint
    this.state = {
      version: {},
      info: [],
    }
    this.t = props.t
  }

  refreshInfo() {
    if (
      this.props.cookie.token === '' ||
      this.props.cookie.token === undefined ||
      (this.props.cookie.token !== 'no_auth' &&
        !sessionStorage.getItem('token'))
    ) {
      return
    }
    fetchWrapper
      .get('/v1/cluster/info?detailed=true')
      .then((data) => {
        const { state } = this
        state['info'] = data
        this.setState(state)
      })
      .catch((error) => {
        console.error('Error:', error)
        if (error.response.status == 403) {
          this.props.handleGoBack()
        }
      })

    if (JSON.stringify(this.state.version) === '{}') {
      fetchWrapper
        .get('/v1/cluster/version')
        .then((data) => {
          const { state } = this
          state['version'] = {
            release: 'v' + data['version'],
            commit: data['full-revisionid'],
          }
          this.setState(state)
        })
        .catch((error) => {
          console.error('Error:', error)
          if (error.response.status == 403) {
            this.props.handleGoBack()
          }
        })
    }
  }

  componentDidMount() {
    this.interval = setInterval(() => this.refreshInfo(), 5000)
    this.refreshInfo()
  }

  componentWillUnmount() {
    clearInterval(this.interval)
  }

  render() {
    if (this.state === undefined || this.state['info'] === []) {
      return <div>Loading</div>
    }

    if (this.nodeRole !== 'Worker-Details') {
      const roleData = this.state['info'].filter(
        (obj) => obj['node_type'] === this.nodeRole
      )

      const sum = (arr) => {
        return arr.reduce((a, b) => a + b, 0)
      }

      const gatherResourceStats = (prop) =>
        sum(roleData.map((obj) => obj[prop]))

      const resourceStats = {
        cpu_total: gatherResourceStats('cpu_count'),
        cpu_avail: gatherResourceStats('cpu_available'),
        memory_total: gatherResourceStats('mem_total'),
        memory_avail: gatherResourceStats('mem_available'),
        gpu_total: gatherResourceStats('gpu_count'),
        gpu_memory_total: gatherResourceStats('gpu_vram_total'),
        gpu_memory_avail: gatherResourceStats('gpu_vram_available'),
      }

      //for all cases, we will at least have cpu information available.
      resourceStats.cpu_used = resourceStats.cpu_total - resourceStats.cpu_avail
      resourceStats.memory_used =
        resourceStats.memory_total - resourceStats.memory_avail

      const row_count = (
        <StyledTableRow>
          <StyledTableCell>{this.t('clusterInfo.count')}</StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid>{roleData.length}</Grid>
            </Grid>
          </StyledTableCell>
        </StyledTableRow>
      )

      const CPU_row = (
        <StyledTableRow>
          <StyledTableCell>{this.t('clusterInfo.cpuInfo')}</StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid xs={4}>
                {this.t('clusterInfo.usage')}{' '}
                {resourceStats.cpu_used.toFixed(2)}
              </Grid>
              <Grid xs={8}>
                {this.t('clusterInfo.total')}{' '}
                {resourceStats.cpu_total.toFixed(2)}
              </Grid>
            </Grid>
          </StyledTableCell>
        </StyledTableRow>
      )

      const CPU_Memory_Info_row = (
        <StyledTableRow>
          <StyledTableCell>
            {this.t('clusterInfo.cpuMemoryInfo')}
          </StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid xs={4}>
                {this.t('clusterInfo.usage')}{' '}
                {toReadableSize(resourceStats.memory_used)}
              </Grid>
              <Grid xs={8}>
                {this.t('clusterInfo.total')}{' '}
                {toReadableSize(resourceStats.memory_total)}
              </Grid>
            </Grid>
          </StyledTableCell>
        </StyledTableRow>
      )

      const version_row = (
        <StyledTableRow>
          <StyledTableCell>{this.t('clusterInfo.version')}</StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid xs={4}>
                {this.t('clusterInfo.release')} {this.state.version.release}
              </Grid>
              <Grid xs={8}>
                {this.t('clusterInfo.commit')} {this.state.version.commit}
              </Grid>
            </Grid>
          </StyledTableCell>
        </StyledTableRow>
      )

      let table_bodies
      //case that we do not have GPU presents.
      if (resourceStats.gpu_memory_total === 0) {
        table_bodies = [row_count, CPU_row, CPU_Memory_Info_row, version_row]
      } else {
        resourceStats.gpu_memory_used =
          resourceStats.gpu_memory_total - resourceStats.gpu_memory_avail

        const GPU_row = (
          <StyledTableRow>
            <StyledTableCell>{this.t('clusterInfo.gpuInfo')}</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={12}>
                  {this.t('clusterInfo.total')}{' '}
                  {resourceStats.gpu_total.toFixed(2)}
                </Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
        )

        const GPU_Memory_Info_row = (
          <StyledTableRow>
            <StyledTableCell>
              {this.t('clusterInfo.gpuMemoryInfo')}
            </StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={4}>
                  {this.t('clusterInfo.usage')}{' '}
                  {toReadableSize(resourceStats.gpu_memory_used)}
                </Grid>
                <Grid xs={8}>
                  {this.t('clusterInfo.total')}{' '}
                  {toReadableSize(resourceStats.gpu_memory_total)}
                </Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
        )

        table_bodies = [
          row_count,
          CPU_row,
          CPU_Memory_Info_row,
          GPU_row,
          GPU_Memory_Info_row,
          version_row,
        ]
      }

      if (this.nodeRole === 'Supervisor') {
        const supervisor_addr_row = (
          <StyledTableRow>
            <StyledTableCell>{this.t('clusterInfo.address')}</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid>{roleData[0] ? roleData[0]['ip_address'] : '-'}</Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
        )
        table_bodies.splice(1, 0, supervisor_addr_row)
      }

      return (
        <Table size="small">
          <TableHead>
            <TableRow>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.item')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                <Grid container>
                  <Grid>{this.t('clusterInfo.value')}</Grid>
                </Grid>
              </StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>{table_bodies}</TableBody>
        </Table>
      )
    } else {
      const workerData = this.state['info'].filter(
        (obj) => obj['node_type'] === 'Worker'
      )

      return (
        <Table size="small">
          <TableHead>
            <TableRow>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.nodeType')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.address')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.cpuUsage')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.cpuTotal')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.memUsage')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.memTotal')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.gpuCount')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.gpuMemUsage')}
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                {this.t('clusterInfo.gpuMemTotal')}
              </StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {workerData.map((row) => (
              <StyledTableRow>
                <StyledTableCell>
                  {this.t('clusterInfo.worker')}
                </StyledTableCell>
                <StyledTableCell>{row['ip_address']}</StyledTableCell>
                <StyledTableCell>
                  {(row['cpu_count'] - row['cpu_available']).toFixed(2)}
                </StyledTableCell>
                <StyledTableCell>{row['cpu_count'].toFixed(2)}</StyledTableCell>
                <StyledTableCell>
                  {toReadableSize(row['mem_total'] - row['mem_available'])}
                </StyledTableCell>
                <StyledTableCell>
                  {toReadableSize(row['mem_total'])}
                </StyledTableCell>
                <StyledTableCell>{row['gpu_count'].toFixed(2)}</StyledTableCell>
                <StyledTableCell>
                  {toReadableSize(
                    row['gpu_vram_total'] - row['gpu_vram_available']
                  )}
                </StyledTableCell>
                <StyledTableCell>
                  {toReadableSize(row['gpu_vram_total'])}
                </StyledTableCell>
              </StyledTableRow>
            ))}
          </TableBody>
        </Table>
      )
    }
  }
}

NodeInfo.propTypes = {
  nodeRole: PropTypes.string,
  endpoint: PropTypes.string,
}

export default NodeInfo
