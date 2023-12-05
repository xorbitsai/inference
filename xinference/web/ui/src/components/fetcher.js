// Some users use https proxies for backend service such as nginx.
// So if the url has a https prefix, add a special header to avoid `Mixed Content` issue.
function updateOptions(url, options) {
  const update = { ...options };
  if (url.startsWith("https://")) {
    update.headers = {
      ...update.headers,
      "Content-Security-Policy": "upgrade-insecure-requests",
    };
  }
  return update;
}

export default function fetcher(url, options) {
  return fetch(url, updateOptions(url, options));
}
