/**
 * Build Grafana iframe URL for the full dashboard (kiosk mode).
 */
export const buildGrafanaUrl = (config, theme = 'light') => {
  const {
    grafana_url,
    grafana_dashboard_uid,
    grafana_datasource,
    cluster_name,
  } = config
  if (!grafana_url) return null

  let url = `${grafana_url}/d/${grafana_dashboard_uid}?orgId=1&kiosk=tv&theme=${theme}`

  if (grafana_datasource) {
    url += `&var-datasource=${encodeURIComponent(grafana_datasource)}`
  }
  if (cluster_name) {
    url += `&var-cluster=${encodeURIComponent(cluster_name)}`
  }

  return url
}

/**
 * Build Grafana single-panel embed URL (for model detail pages).
 */
export const buildGrafanaPanelUrl = (
  config,
  panelId,
  modelName,
  theme = 'light'
) => {
  const {
    grafana_url,
    grafana_dashboard_uid,
    grafana_datasource,
    cluster_name,
  } = config
  if (!grafana_url) return null

  let url = `${grafana_url}/d-solo/${grafana_dashboard_uid}?orgId=1&theme=${theme}&panelId=${panelId}`

  if (grafana_datasource) {
    url += `&var-datasource=${encodeURIComponent(grafana_datasource)}`
  }
  if (cluster_name) {
    url += `&var-cluster=${encodeURIComponent(cluster_name)}`
  }
  if (modelName) {
    url += `&var-model_name=${encodeURIComponent(modelName)}`
  }

  return url
}
