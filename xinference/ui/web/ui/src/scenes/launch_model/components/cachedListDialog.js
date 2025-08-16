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

const EditCustomModel = ({ open, modelData, modelType, onClose, onUpdate }) => {
  const { t } = useTranslation()
  const { setErrorMsg } = useContext(ApiContext)
  const [cachedListArr, setCachedListArr] = useState([])
  const [page, setPage] = useState(0)
  const [isDeleteCached, setIsDeleteCached] = useState(false)
  const [cachedModelVersion, setCachedModelVersion] = useState('')

  const StyledTableRow = styled(TableRow)(({ theme }) => ({
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.action.hover,
    },
  }))

  const emptyRows =
    page >= 0 ? Math.max(0, (1 + page) * 5 - cachedListArr.length) : 0

  const getCachedList = () => {
    fetchWrapper
      .get(`/v1/cache/models?model_name=${modelData.model_name}`)
      .then((data) => {
        setCachedListArr(data.list)
        if (
          page !== 0 &&
          data.list.length &&
          (page + 1) * 5 >= cachedListArr.length &&
          data.list.length % 5 === 0
        ) {
          setPage(data.list.length / 5 - 1)
        }
      })
      .catch((error) => {
        console.error(error)
        if (error.response.status !== 403) {
          setErrorMsg(error.message)
        }
      })
  }

  const handleDeleteCached = () => {
    fetchWrapper
      .delete(`/v1/cache/models?model_version=${cachedModelVersion}`)
      .then(() => {
        getCachedList()
        setIsDeleteCached(false)
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

  const handleOpenDeleteCachedDialog = (model_version) => {
    setCachedModelVersion(model_version)
    setIsDeleteCached(true)
  }

  const handleCloseCachedList = () => {
    onClose()
    onUpdate()
  }

  useEffect(() => {
    if (open) getCachedList()
  }, [open])

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xl"
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle sx={{ m: 0, p: 2 }} id="customized-dialog-title">
        {modelData.model_name}
      </DialogTitle>
      <Box
        sx={(theme) => ({
          position: 'absolute',
          right: 8,
          top: 8,
          color: theme.palette.grey[500],
        })}
      >
        <Close style={{ cursor: 'pointer' }} onClick={handleCloseCachedList} />
      </Box>
      <DialogContent>
        <TableContainer component={Paper}>
          <Table
            sx={{ minWidth: 500 }}
            style={{ height: '500px', width: '100%' }}
            stickyHeader
            aria-label="simple pagination table"
          >
            <TableHead>
              <TableRow>
                {modelType === 'LLM' && (
                  <>
                    <TableCell align="left">
                      {t('launchModel.model_format')}
                    </TableCell>
                    <TableCell align="left">
                      {t('launchModel.model_size_in_billions')}
                    </TableCell>
                    <TableCell align="left">
                      {t('launchModel.quantizations')}
                    </TableCell>
                  </>
                )}
                <TableCell align="left" style={{ width: 192 }}>
                  {t('launchModel.real_path')}
                </TableCell>
                <TableCell align="left" style={{ width: 46 }}></TableCell>
                <TableCell align="left" style={{ width: 192 }}>
                  {t('launchModel.path')}
                </TableCell>
                <TableCell align="left" style={{ width: 46 }}></TableCell>
                <TableCell
                  align="left"
                  style={{ whiteSpace: 'nowrap', minWidth: 116 }}
                >
                  {t('launchModel.ipAddress')}
                </TableCell>
                <TableCell align="left">{t('launchModel.operation')}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody style={{ position: 'relative' }}>
              {cachedListArr.slice(page * 5, page * 5 + 5).map((row) => (
                <StyledTableRow style={{ maxHeight: 90 }} key={row.model_name}>
                  {modelType === 'LLM' && (
                    <>
                      <TableCell component="th" scope="row">
                        {row.model_format === null ? '—' : row.model_format}
                      </TableCell>
                      <TableCell>
                        {row.model_size_in_billions === null
                          ? '—'
                          : row.model_size_in_billions}
                      </TableCell>
                      <TableCell>
                        {row.quantization === null ? '—' : row.quantization}
                      </TableCell>
                    </>
                  )}
                  <TableCell>
                    <Tooltip title={row.real_path}>
                      <div
                        className={
                          modelType === 'LLM' ? 'pathBox' : 'pathBox pathBox2'
                        }
                      >
                        {row.real_path}
                      </div>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <CopyComponent
                      tip={t('launchModel.copyRealPath')}
                      text={row.real_path}
                    />
                  </TableCell>
                  <TableCell>
                    <Tooltip title={row.path}>
                      <div
                        className={
                          modelType === 'LLM' ? 'pathBox' : 'pathBox pathBox2'
                        }
                      >
                        {row.path}
                      </div>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <CopyComponent
                      tip={t('launchModel.copyPath')}
                      text={row.path}
                    />
                  </TableCell>
                  <TableCell>{row.actor_ip_address}</TableCell>
                  <TableCell align={modelType === 'LLM' ? 'center' : 'left'}>
                    <IconButton
                      aria-label="delete"
                      size="large"
                      onClick={() =>
                        handleOpenDeleteCachedDialog(row.model_version)
                      }
                    >
                      <Delete />
                    </IconButton>
                  </TableCell>
                </StyledTableRow>
              ))}
              {emptyRows > 0 && (
                <TableRow style={{ height: 89.4 * emptyRows }}>
                  <TableCell />
                </TableRow>
              )}
              {cachedListArr.length === 0 && (
                <div className="empty">{t('launchModel.noCacheForNow')}</div>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          style={{ float: 'right' }}
          rowsPerPageOptions={[5]}
          count={cachedListArr.length}
          rowsPerPage={5}
          page={page}
          onPageChange={handleChangePage}
        />
      </DialogContent>

      <DeleteDialog
        text={t('launchModel.confirmDeleteCacheFiles')}
        isDelete={isDeleteCached}
        onHandleIsDelete={() => setIsDeleteCached(false)}
        onHandleDelete={handleDeleteCached}
      />
    </Dialog>
  )
}

export default EditCustomModel
