const getEndpoint = () => {
  let endPoint = ''
  if (!process.env.NODE_ENV || process.env.NODE_ENV === 'development') {
    endPoint = 'http://127.0.0.1:9997'
  } else {
    const fullUrl = window.location.href
    endPoint = fullUrl.split('/ui')[0]
  }
  return endPoint
}

const isValidBearerToken = (token) => {
  return (
    token !== '' && token !== undefined && token !== null && token.length > 10
  )
}

export { getEndpoint, isValidBearerToken }
