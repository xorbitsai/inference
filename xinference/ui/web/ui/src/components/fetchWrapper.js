import { Cookies } from 'react-cookie'

const cookies = new Cookies()

const getBaseUrl = () => {
  let base_URL = ''
  if (!process.env.NODE_ENV || process.env.NODE_ENV === 'development') {
    base_URL = 'http://127.0.0.1:9997'
  } else {
    const fullUrl = window.location.href
    base_URL = fullUrl.split('/ui')[0]
  }
  return base_URL
}

const apiBase = getBaseUrl()

let isRefreshing = false
let refreshPromise = null
let failedQueue = []

const processQueue = (success) => {
  failedQueue.forEach((prom) => {
    if (success) {
      prom.resolve()
    } else {
      prom.reject()
    }
  })
  failedQueue = []
}

const tryRefreshToken = async () => {
  if (isRefreshing) {
    return new Promise((resolve, reject) => {
      failedQueue.push({ resolve, reject })
    })
  }
  const refreshToken = sessionStorage.getItem('refresh_token')
  if (!refreshToken) return false
  isRefreshing = true
  refreshPromise = fetch(`${apiBase}/v1/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
  })
    .then(async (res) => {
      if (res.ok) {
        const data = await res.json()
        cookies.set('token', data.access_token, { path: '/' })
        sessionStorage.setItem('token', data.access_token)
        if (data.refresh_token) {
          sessionStorage.setItem('refresh_token', data.refresh_token)
        }
        processQueue(true)
        return true
      }
      processQueue(false)
      return false
    })
    .catch(() => {
      processQueue(false)
      return false
    })
    .finally(() => {
      isRefreshing = false
      refreshPromise = null
    })
  return refreshPromise
}

const getAuthHeaders = () => {
  const headers = {}
  if (
    cookies.get('token') !== 'no_auth' &&
    sessionStorage.getItem('token') !== 'no_auth'
  ) {
    headers.Authorization = 'Bearer ' + sessionStorage.getItem('token')
  }
  return headers
}

const fetchWithRetry = async (url, options) => {
  const response = await fetch(url, options)
  if (response.status === 401 && sessionStorage.getItem('refresh_token')) {
    const refreshed = await tryRefreshToken()
    if (refreshed) {
      const newHeaders = {
        ...options.headers,
        Authorization: 'Bearer ' + sessionStorage.getItem('token'),
      }
      return fetch(url, { ...options, headers: newHeaders })
    }
  }
  return response
}

const fetchWrapper = {
  get: async (endpoint, config = {}) => {
    const url = `${apiBase}${endpoint}`
    const headers = {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...config.headers,
    }
    const response = await fetchWithRetry(url, {
      method: 'GET',
      ...config,
      headers,
    })
    return fetchWrapper.handleResponse(response)
  },

  post: async (endpoint, body, config = {}) => {
    const url = `${apiBase}${endpoint}`
    const headers = {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...config.headers,
    }
    const response = await fetchWithRetry(url, {
      method: 'POST',
      body: JSON.stringify(body),
      ...config,
      headers,
    })
    return fetchWrapper.handleResponse(response)
  },

  put: async (endpoint, body, config = {}) => {
    const url = `${apiBase}${endpoint}`
    const headers = {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...config.headers,
    }
    const response = await fetchWithRetry(url, {
      method: 'PUT',
      body: JSON.stringify(body),
      ...config,
      headers,
    })
    return fetchWrapper.handleResponse(response)
  },

  delete: async (endpoint, config = {}) => {
    const url = `${apiBase}${endpoint}`
    const headers = {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...config.headers,
    }
    const response = await fetchWithRetry(url, {
      method: 'DELETE',
      ...config,
      headers,
    })
    return fetchWrapper.handleResponse(response)
  },

  handleResponse: async (response) => {
    if (
      response.status == 401 &&
      localStorage.getItem('authStatus') !== '401'
    ) {
      localStorage.setItem('authStatus', '401')
      window.dispatchEvent(new Event('auth-status'))
    } else if (
      response.status == 403 &&
      localStorage.getItem('authStatus') !== '403'
    ) {
      localStorage.setItem('authStatus', '403')
      window.dispatchEvent(new Event('auth-status'))
    }

    if (!response.ok) {
      const errorData = await response.json()
      const error = new Error(
        `Server error: ${response.status} - ${
          errorData.detail || 'Unknown error'
        }`
      )
      error.response = response
      throw error
    }
    const data = await response.json()
    return data
  },
}

export default fetchWrapper
