import { Cookies } from 'react-cookie'

const cookies = new Cookies()

const updateOptions = (url, options) => {
  const update = { ...options }
  if (
    cookies.get('token') !== 'no_auth' &&
    sessionStorage.getItem('token') !== 'no_auth'
  ) {
    update.headers = {
      ...update.headers,
      Authorization: 'Bearer ' + sessionStorage.getItem('token'),
    }
  }
  return update
}

export default function fetcher(url, options) {
  return fetch(url, updateOptions(url, options)).then((res) => {
    // For the situation that server has already been restarted, the current token may become invalid,
    // which leads to UI hangs.
    if (res.status == 401 && localStorage.getItem('authStatus') !== '401') {
      localStorage.setItem('authStatus', '401')
      window.dispatchEvent(new Event('auth-status'))
    } else if (
      res.status == 403 &&
      localStorage.getItem('authStatus') !== '403'
    ) {
      localStorage.setItem('authStatus', '403')
      window.dispatchEvent(new Event('auth-status'))
    } else {
      return res
    }
  })
}
