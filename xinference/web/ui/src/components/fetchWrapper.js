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

const fetchWrapper = {
  get: async (endpoint, config = {}) => {
    const url = `${apiBase}${endpoint}`
    const headers = {
      'Content-Type': 'application/json',
      ...config.headers,
    }
    if (cookies.get('token') !== 'no_auth' && sessionStorage.getItem('token') !== 'no_auth') {
      headers.Authorization = 'Bearer ' + sessionStorage.getItem('token')
    }
    const response = await fetch(url, {
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
      ...config.headers,
    }
    if (cookies.get('token') !== 'no_auth' && sessionStorage.getItem('token') !== 'no_auth') {
      headers.Authorization = 'Bearer ' + sessionStorage.getItem('token')
    }
    const response = await fetch(url, {
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
      ...config.headers,
    }
    if (cookies.get('token') !== 'no_auth' && sessionStorage.getItem('token') !== 'no_auth') {
      headers.Authorization = 'Bearer ' + sessionStorage.getItem('token')
    }
    const response = await fetch(url, {
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
      ...config.headers,
    }
    if (cookies.get('token') !== 'no_auth' && sessionStorage.getItem('token') !== 'no_auth') {
      headers.Authorization = 'Bearer ' + sessionStorage.getItem('token')
    }
    const response = await fetch(url, {
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
