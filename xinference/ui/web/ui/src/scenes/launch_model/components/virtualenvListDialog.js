import { Close, Delete } from '@mui/icons-material'
import {
  Box,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  Tooltip,
} from '@mui/material'
import { styled } from '@mui/material/styles'
import React, { useContext, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ApiContext } from '../../../components/apiContext'
import CopyComponent from '../../../components/copyComponent'
import DeleteDialog from '../../../components/deleteDialog'
import fetchWrapper from '../../../components/fetchWrapper'

const VirtualEnvListDialog = ({ open, onClose, onUpdate, modelData }) => {
  const { t } = useTranslation()
  const { setErrorMsg } = useContext(ApiContext)
  const [virtualenvListArr, setVirtualenvListArr] = useState([])
  const [page, setPage] = useState(0)
  const [isDeleteVirtualenv, setIsDeleteVirtualenv] = useState(false)
  const [selectedModelName, setSelectedModelName] = useState('')

  const StyledTableRow = styled(TableRow)(({ theme }) => ({
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.action.hover,
    },
  }))

  const emptyRows =
    page >= 0 ? Math.max(0, (1 + page) * 5 - virtualenvListArr.length) : 0

  const getVirtualenvList = () => {
    if (!modelData?.model_name) return

    fetchWrapper
      .get(`/v1/virtualenvs?model_name=${modelData.model_name}`)
      .then((data) => {
        setVirtualenvListArr(data.list)
        // Always reset to page 0 since we only show one model's virtual environment
        setPage(0)
      })
      .catch((error) => {
        if (error.response.status !== 403) {
          setErrorMsg(error.message)
        }
      })
  }

  const handleDeleteVirtualenv = () => {
    fetchWrapper
      .delete(`/v1/virtualenvs?model_name=${selectedModelName}`)
      .then(() => {
        getVirtualenvList()
        setIsDeleteVirtualenv(false)
        setSelectedModelName('')
      })
      .catch((error) => {
        console.error(error)
        if (error.response.status !== 403) {
          setErrorMsg(error.message)
        }
      })
  }

  const handleChangePage = (_, newPage) => {
    setPage(newPage)
  }

  const handleOpenDeleteVirtualenvDialog = (modelName) => {
    setSelectedModelName(modelName)
    setIsDeleteVirtualenv(true)
  }

  const handleCloseVirtualenvList = () => {
    onClose()
    onUpdate()
  }

  useEffect(() => {
    if (open) getVirtualenvList()
  }, [open])

  return (
    <>
      <Dialog
        open={open}
        onClose={onClose}
        maxWidth="xl"
        aria-labelledby="virtualenv-dialog-title"
        aria-describedby="virtualenv-dialog-description"
      >
        <DialogTitle sx={{ m: 0, p: 2 }} id="customized-dialog-title">
          {modelData?.model_name || t('launchModel.manageVirtualEnvironments')}
        </DialogTitle>
        <Box
          sx={(theme) => ({
            position: 'absolute',
            right: 8,
            top: 8,
            color: theme.palette.grey[500],
          })}
        >
          <Close
            style={{ cursor: 'pointer' }}
            onClick={handleCloseVirtualenvList}
          />
        </Box>
        <DialogContent>
          <TableContainer component={Paper}>
            <Table
              sx={{ minWidth: 700 }}
              style={{ height: '500px', width: '100%' }}
              stickyHeader
              aria-label="virtualenv pagination table"
            >
              <TableHead>
                <TableRow>
                  <TableCell align="left">
                    {t('launchModel.modelName')}
                  </TableCell>
                  <TableCell align="left">{t('launchModel.envPath')}</TableCell>
                  <TableCell align="left" style={{ width: 46 }}></TableCell>
                  <TableCell align="left">
                    {t('launchModel.pythonVersion')}
                  </TableCell>
                  <TableCell align="left">
                    {t('launchModel.ipAddress')}
                  </TableCell>
                  <TableCell align="left">
                    {t('launchModel.operation')}
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody style={{ position: 'relative' }}>
                {virtualenvListArr.slice(page * 5, page * 5 + 5).map((row) => (
                  <StyledTableRow
                    style={{ maxHeight: 90 }}
                    key={
                      row.path ||
                      `${row.model_name}-${row.model_engine || 'default'}-${
                        row.python_version || 'unknown'
                      }`
                    }
                  >
                    <TableCell component="th" scope="row">
                      <Tooltip title={row.model_name}>
                        <div className="pathBox" style={{ maxWidth: 150 }}>
                          {row.model_name}
                        </div>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <Tooltip title={row.path}>
                        <div className="pathBox" style={{ maxWidth: 200 }}>
                          {row.path}
                        </div>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <CopyComponent
                        tip={t('launchModel.copyEnvPath')}
                        text={row.path}
                      />
                    </TableCell>
                    <TableCell>{row.python_version || '—'}</TableCell>
                    <TableCell>{row.actor_ip_address}</TableCell>
                    <TableCell align="center">
                      <Tooltip title={t('launchModel.delete')}>
                        <IconButton
                          aria-label="delete"
                          size="large"
                          onClick={() =>
                            handleOpenDeleteVirtualenvDialog(row.model_name)
                          }
                        >
                          <Delete />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </StyledTableRow>
                ))}
                {emptyRows > 0 && (
                  <TableRow style={{ height: 89.4 * emptyRows }}>
                    <TableCell />
                  </TableRow>
                )}
                {virtualenvListArr.length === 0 && (
                  <div className="empty">
                    {t('launchModel.noVirtualEnvironmentsForNow')}
                  </div>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            style={{ float: 'right' }}
            rowsPerPageOptions={[5]}
            count={virtualenvListArr.length}
            rowsPerPage={5}
            page={page}
            onPageChange={handleChangePage}
          />
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <DeleteDialog
        text={t('launchModel.confirmDeleteVirtualEnv')}
        isDelete={isDeleteVirtualenv}
        onHandleIsDelete={() => setIsDeleteVirtualenv(false)}
        onHandleDelete={handleDeleteVirtualenv}
      />
    </>
  )
}

export default VirtualEnvListDialog
