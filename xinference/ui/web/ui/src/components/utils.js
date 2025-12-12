export const getEndpoint = () => {
  let endPoint = ''
  if (!process.env.NODE_ENV || process.env.NODE_ENV === 'development') {
    endPoint = 'http://127.0.0.1:9997'
  } else {
    const fullUrl = window.location.href
    endPoint = fullUrl.split('/ui')[0]
  }
  return endPoint
}

export const isValidBearerToken = (token) => {
  return (
    token !== '' && token !== undefined && token !== null && token.length > 10
  )
}

export const toReadableSize = (size) => {
  const res_size = size / 1024.0 ** 2
  return res_size.toFixed(2) + 'MiB'
}

export async function copyToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      return false
    }
  }

  try {
    const container = document.querySelector('[role="dialog"]') ||
      document.querySelector('.drawerCard') ||
      document.body;

    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'absolute';
    textArea.style.left = '-9999px';
    container.appendChild(textArea);

    textArea.select();
    const success = document.execCommand('copy');

    container.removeChild(textArea);
    return success;
  } catch {
    return false;
  }
}
