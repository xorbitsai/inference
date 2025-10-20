import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
} from '@mui/material'
import React, { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

const API_BASE_URL = 'https://model.xinference.io'

const AddModelDialog = ({ open, onClose }) => {
  const { t } = useTranslation()
  const [url, setUrl] = useState('')
  const [loginOpen, setLoginOpen] = useState(false)
  const [pendingModelId, setPendingModelId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const loginIframeRef = useRef(null)

  const handleClose = (type) => {
    setErrorMsg('')

    const actions = {
      add: onClose,
      login: () => setLoginOpen(false),
    }

    actions[type]?.()
  }

  const extractModelId = (input) => {
    try {
      const u = new URL(input)
      const m1 = u.pathname.match(/\/(\d+)(?:\/?$)/)
      if (m1 && m1[1]) return m1[1]
      const qp = u.searchParams.get('model_id')
      if (qp) return qp
    } catch (e) {
      const m2 = String(input).match(/(\d+)(?:\/?$)/)
      if (m2 && m2[1]) return m2[1]
    }
    return null
  }

  // 修改：download 默认从 sessionStorage 读取 token（若传参提供则优先）
  // performDownload：收到 token 后直连接口，获取 JSON
  const performDownload = async (
    modelId,
    tokenFromParam,
    fromLogin = false
  ) => {
    const endpoint = `${API_BASE_URL}/api/models/download?model_id=${encodeURIComponent(
      modelId
    )}`
    const effectiveToken =
      tokenFromParam ||
      sessionStorage.getItem('model_hub_token') ||
      localStorage.getItem('io_login_success')
    const headers = effectiveToken
      ? { Authorization: `Bearer ${effectiveToken}` }
      : {}
    setLoading(true)
    setErrorMsg('')
    try {
      const res = await fetch(endpoint, {
        method: 'GET',
        headers,
      })

      if (res.status === 401) {
        const refreshToken = sessionStorage.getItem('model_hub_refresh_token')
        if (!refreshToken) {
          sessionStorage.removeItem('model_hub_token')
          setPendingModelId(modelId)
          setLoginOpen(true)
          return
        }
        try {
          const refreshRes = await fetch(`${API_BASE_URL}/api/users/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: refreshToken }),
          })
          if (!refreshRes.ok) {
            throw new Error(`refresh failed: ${refreshRes.status}`)
          }
          const refreshData = await refreshRes.json().catch(() => ({}))
          const newToken = refreshData?.data?.accessToken
          if (newToken) {
            sessionStorage.setItem('model_hub_token', newToken)
            await performDownload(modelId, newToken, false)
            return
          } else {
            sessionStorage.removeItem('model_hub_token')
            setPendingModelId(modelId)
            setLoginOpen(true)
            return
          }
        } catch (e) {
          sessionStorage.removeItem('model_hub_token')
          setPendingModelId(modelId)
          setLoginOpen(true)
          return
        }
      }

      if (res.status === 403) {
        let detailMsg = ''
        try {
          const body = await res.json()
          if (body?.error_code === 'MODEL_PRIVATE') {
            detailMsg = t('launchModel.error.modelPrivate')
          } else if (body?.message) {
            detailMsg = body.message
          }
        } catch {
          console.log('')
        }
        if (fromLogin) {
          setErrorMsg(
            detailMsg || t('launchModel.error.noPermissionAfterLogin')
          )
          return
        } else {
          setPendingModelId(modelId)
          setLoginOpen(true)
          return
        }
      }

      if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new Error(
          t('launchModel.error.downloadFailed', { status: res.status, text })
        )
      }
      const data = await res.json()
      console.log('models/download 响应:', data)
      handleClose('add')
    } catch (err) {
      console.error(err)
      setErrorMsg(err.message || t('launchModel.error.requestFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleFormSubmit = async (e) => {
    e.preventDefault()
    const modelId = extractModelId(url?.trim())
    if (!modelId) {
      setErrorMsg(t('launchModel.error.cannotExtractModelId'))
      return
    }
    await performDownload(modelId)
  }

  useEffect(() => {
    const listener = (event) => {
      if (event.origin !== API_BASE_URL) return
      const { type, token, refresh_token } = event.data || {}

      if (type === 'io_login_success' && token && refresh_token) {
        handleClose('login')
        sessionStorage.setItem('model_hub_token', token)
        sessionStorage.setItem('model_hub_refresh_token', refresh_token)
        if (pendingModelId) {
          void performDownload(pendingModelId, token, true)
        }
      }
    }

    window.addEventListener('message', listener)
    return () => {
      window.removeEventListener('message', listener)
    }
  }, [pendingModelId])

  return (
    <Dialog open={open} onClose={() => handleClose('add')} width={500}>
      <DialogTitle>{t('launchModel.addModel')}</DialogTitle>
      <DialogContent>
        <div
          style={{
            width: '500px',
            minHeight: '160px',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}
        >
          <div>
            {t('launchModel.addModelDialog.introPrefix')}{' '}
            <a
              href={`${API_BASE_URL}/models`}
              target="_blank"
              rel="noopener noreferrer"
              style={{ textDecoration: 'none', color: '#1976d2' }}
            >
              {t('launchModel.addModelDialog.platformLinkText')}
            </a>
            {t('launchModel.addModelDialog.introSuffix')}
          </div>
          <div>
            {t('launchModel.addModelDialog.example', {
              modelName: 'qwen3',
              modelUrl: 'https://model.xinference.io/models/detail/250',
            })}
          </div>
          <form onSubmit={handleFormSubmit} id="subscription-form">
            <TextField
              autoFocus
              required
              margin="dense"
              id="url"
              name="url"
              label={t('launchModel.addModelDialog.urlLabel')}
              fullWidth
              placeholder={t('launchModel.placeholderTip')}
              value={url}
              onChange={(e) => {
                setUrl(e.target.value)
              }}
              disabled={loading}
            />
          </form>
          {errorMsg && <div style={{ color: '#d32f2f' }}>{errorMsg}</div>}
        </div>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => handleClose('add')} disabled={loading}>
          {t('launchModel.cancel')}
        </Button>
        <Button
          autoFocus
          type="submit"
          form="subscription-form"
          disabled={loading}
        >
          {t('launchModel.confirm')}
        </Button>
      </DialogActions>

      <Dialog open={loginOpen} onClose={() => handleClose('login')}>
        <div
          style={{
            width: '100%',
            maxWidth: 640,
            padding: 16,
            boxSizing: 'border-box',
          }}
        >
          <iframe
            ref={loginIframeRef}
            src={`${API_BASE_URL}/signin`}
            title="Model Platform Signin"
            style={{ width: '100%', minHeight: 520, border: 0 }}
          />
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              marginTop: 12,
            }}
          >
            <Button onClick={() => handleClose('login')} disabled={loading}>
              关闭
            </Button>
          </div>
        </div>
      </Dialog>
    </Dialog>
  )
}

export default AddModelDialog
