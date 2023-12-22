import { Cookies } from 'react-cookie'

const cookies = new Cookies()

const updateOptions = (url, options) => {
  const update = { ...options }
  if (cookies.get('token') !== 'no_auth') {
    update.headers = {
      ...update.headers,
      Authorization: 'Bearer ' + cookies.get('token'),
    }
  }
  return update
}

export default function fetcher(url, options) {
  return fetch(url, updateOptions(url, options)).then((res) => {
    // For the situation that server has already been restarted, the current token may become invalid,
    // which leads to UI hangs.
    if (
      res.status === 401 &&
      cookies.get('token') &&
      cookies.get('token').length > 10 // TODO: more reasonable token check
    ) {
      cookies.remove('token', { path: '/' })
      window.location.href = '/ui/#/login'
    } else {
      return res
    }
  })
}
