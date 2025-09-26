
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    TextField
} from '@mui/material'
import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'

const API_BASE_URL = 'https://model.xinference.io'

const AddModelDialog = ({ open, onClose }) => {
    const { t } = useTranslation()
    const [url, setUrl] = useState('')
    const [loginOpen, setLoginOpen] = useState(false)
    const [usernameOrEmail, setUsernameOrEmail] = useState('')
    const [password, setPassword] = useState('')
    const [pendingModelId, setPendingModelId] = useState(null)
    const [loading, setLoading] = useState(false)
    const [errorMsg, setErrorMsg] = useState('')

    const handleClose = (type) => {
        setErrorMsg('');
      
        const actions = {
          add: onClose,
          login: () => setLoginOpen(false),
        };
      
        actions[type]?.();
    };      

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

    const performDownload = async (modelId, token, fromLogin = false) => {
        const endpoint = `${API_BASE_URL}/api/models/download?model_id=${encodeURIComponent(modelId)}`
        const headers = token ? { Authorization: `Bearer ${token}` } : {}
        setLoading(true)
        setErrorMsg('')
        try {
            const res = await fetch(endpoint, {
                method: 'GET',
                headers,
            })
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
                    // ignore and use default message
                }

                if (fromLogin) {
                    setErrorMsg(detailMsg || t('launchModel.error.noPermissionAfterLogin'))
                    return
                } else {
                    setPendingModelId(modelId)
                    setLoginOpen(true)
                    return
                }
            }
            if (!res.ok) {
                const text = await res.text().catch(() => '')
                throw new Error(t('launchModel.error.downloadFailed', { status: res.status, text }))
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

    const handleLoginSubmit = async (e) => {
        e.preventDefault()
        if (!pendingModelId) return
        setLoading(true)
        setErrorMsg('')
        try {
            const loginRes = await fetch(`${API_BASE_URL}/api/users/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    usernameOrEmail: usernameOrEmail.trim(),
                    password: password,
                }),
            })
            if (!loginRes.ok) {
                const text = await loginRes.text().catch(() => '')
                throw new Error(t('launchModel.error.loginFailedText', { status: loginRes.status, text }))
            }
            const loginJson = await loginRes.json()
            const token = loginJson.data?.accessToken
            if (!token) {
                throw new Error(t('launchModel.error.noTokenAfterLogin'))
            }
            handleClose('login')
            await performDownload(pendingModelId, token, true)
        } catch (err) {
            console.error(err)
            setErrorMsg(err.message || t('launchModel.error.requestFailed'))
        } finally {
            setLoading(false)
        }
    }

    return (
        <Dialog open={open} onClose={() => handleClose('add')} width={500}>
            <DialogTitle>{t('launchModel.addModel')}</DialogTitle>
            <DialogContent>
                <div style={{ width: '500px', minHeight: '160px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div>
                        {t('launchModel.addModelDialog.introPrefix')}{' '}
                        <a
                            href='https://model.xinference.io/models'
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{textDecoration: 'none', color: '#1976d2'}}
                        >
                            {t('launchModel.addModelDialog.platformLinkText')}
                        </a>
                        {t('launchModel.addModelDialog.introSuffix')}
                    </div>
                    <div>
                        {t('launchModel.addModelDialog.example', {
                            modelName: 'qwen3',
                            modelUrl: 'https://model.xinference.io/models/detail/250'
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
                <Button onClick={() => handleClose('add')} disabled={loading}>{t('launchModel.cancel')}</Button>
                <Button autoFocus type="submit" form="subscription-form" disabled={loading}>
                    {t('launchModel.confirm')}
                </Button>
            </DialogActions>

            {/* 403 */}
            <Dialog open={loginOpen} onClose={() => handleClose('login')}>
                <DialogTitle>{t('launchModel.loginDialog.title')}</DialogTitle>
                <DialogContent>
                    <form onSubmit={handleLoginSubmit} id="login-form" style={{ display: 'flex', flexDirection: 'column', gap: 12, width: 360, paddingTop: 10 }}>
                        <TextField
                            required
                            id="usernameOrEmail"
                            name="usernameOrEmail"
                            label={t('launchModel.loginDialog.usernameOrEmail')}
                            fullWidth
                            value={usernameOrEmail}
                            onChange={(e) => setUsernameOrEmail(e.target.value)}
                            disabled={loading}
                        />
                        <TextField
                            required
                            id="password"
                            name="password"
                            label={t('launchModel.loginDialog.password')}
                            type="password"
                            fullWidth
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            disabled={loading}
                        />
                    </form>
                    {errorMsg && <div style={{ color: '#d32f2f', marginTop: 8 }}>{errorMsg}</div>}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => handleClose('login')} disabled={loading}>{t('launchModel.cancel')}</Button>
                    <Button type="submit" form="login-form" disabled={loading}>{t('launchModel.loginDialog.login')}</Button>
                </DialogActions>
            </Dialog>
        </Dialog>
    )
}

export default AddModelDialog