import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Grid from '@mui/material/Unstable_Grid2'
import PropTypes from 'prop-types'
import React from 'react'

import fetcher from '../../components/fetcher'
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
  }

  refreshInfo() {
    fetcher(`${this.endpoint}/v1/cluster/info?detailed=true`, { method: 'GET' })
      .then((res) => {
        if (!res.ok) {
          res.json().then((errorData) => {
            if (errorData.detail === 'Not enough permissions') {
              console.log('Not enough permissions')
              // window.history.back();
            }
          })
        }
        if (res.ok) {
          res.json().then((data) => {
            const { state } = this
            state['info'] = data
            this.setState(state)
          })
        }
      })
      .catch((err) => {
        console.error('Error:', err)
      })

    if (JSON.stringify(this.state.version) === '{}') {
      fetcher(`${this.endpoint}/v1/cluster/version`, {
        method: 'GET',
      })
        .then((res) => {
          if (res.ok) {
            res.json().then((data) => {
              const { state } = this
              state['version'] = {
                release: 'v' + data['version'],
                commit: data['full-revisionid'],
              }
              this.setState(state)
            })
          }
        })
        .catch((err) => {
          console.error('Error:', err)
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
          <StyledTableCell>Count</StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid>{roleData.length}</Grid>
            </Grid>
          </StyledTableCell>
        </StyledTableRow>
      )

      const CPU_row = (
        <StyledTableRow>
          <StyledTableCell>CPU Info</StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid xs={4}>Usage: {resourceStats.cpu_used.toFixed(2)}</Grid>
              <Grid xs={8}>Total: {resourceStats.cpu_total.toFixed(2)}</Grid>
            </Grid>
          </StyledTableCell>
        </StyledTableRow>
      )

      const CPU_Memory_Info_row = (
        <StyledTableRow>
          <StyledTableCell>CPU Memory Info</StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid xs={4}>
                Usage: {toReadableSize(resourceStats.memory_used)}
              </Grid>
              <Grid xs={8}>
                Total: {toReadableSize(resourceStats.memory_total)}
              </Grid>
            </Grid>
          </StyledTableCell>
        </StyledTableRow>
      )

      const version_row = (
        <StyledTableRow>
          <StyledTableCell>Version</StyledTableCell>
          <StyledTableCell>
            <Grid container>
              <Grid xs={4}>Release: {this.state.version.release}</Grid>
              <Grid xs={8}>Commit: {this.state.version.commit}</Grid>
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
            <StyledTableCell>GPU Info</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={12}>Total: {resourceStats.gpu_total.toFixed(2)}</Grid>
              </Grid>
            </StyledTableCell>
          </StyledTableRow>
        )

        const GPU_Memory_Info_row = (
          <StyledTableRow>
            <StyledTableCell>GPU Memory Info</StyledTableCell>
            <StyledTableCell>
              <Grid container>
                <Grid xs={4}>
                  Usage: {toReadableSize(resourceStats.gpu_memory_used)}
                </Grid>
                <Grid xs={8}>
                  Total: {toReadableSize(resourceStats.gpu_memory_total)}
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
            <StyledTableCell>Address</StyledTableCell>
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
                Item
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                <Grid container>
                  <Grid>Value</Grid>
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
                Node Type
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                Address
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                CPU Usage
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                CPU Total
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                Mem Usage
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                Mem Total
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                GPU Count
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                GPU Mem Usage
              </StyledTableCell>
              <StyledTableCell style={{ fontWeight: 'bolder' }}>
                GPU Mem Total
              </StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {workerData.map((row) => (
              <StyledTableRow>
                <StyledTableCell>Worker</StyledTableCell>
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
