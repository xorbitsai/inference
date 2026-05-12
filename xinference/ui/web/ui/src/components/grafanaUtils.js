/**
 * Build Grafana iframe URL for the full dashboard (kiosk mode).
 */
export const buildGrafanaUrl = (
  config,
  theme = 'light',
  from = 'now-1h',
  to = 'now'
) => {
  const {
    grafana_url,
    grafana_dashboard_uid,
    grafana_datasource,
    grafana_alert_datasource,
    cluster_name,
  } = config
  if (!grafana_url) return null

  let url = `${grafana_url}/d/${grafana_dashboard_uid}?orgId=1&kiosk&theme=${theme}&from=${from}&to=${to}`

  if (grafana_datasource) {
    url += `&var-datasource=${encodeURIComponent(grafana_datasource)}`
  }
  if (grafana_alert_datasource) {
    url += `&var-alert_datasource=${encodeURIComponent(
      grafana_alert_datasource
    )}`
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
  theme = 'light',
  from = 'now-1h',
  to = 'now'
) => {
  const {
    grafana_url,
    grafana_dashboard_uid,
    grafana_datasource,
    grafana_alert_datasource,
    cluster_name,
  } = config
  if (!grafana_url) return null

  let url = `${grafana_url}/d-solo/${grafana_dashboard_uid}?orgId=1&kiosk&theme=${theme}&panelId=${panelId}&from=${from}&to=${to}`

  if (grafana_datasource) {
    url += `&var-datasource=${encodeURIComponent(grafana_datasource)}`
  }
  if (grafana_alert_datasource) {
    url += `&var-alert_datasource=${encodeURIComponent(
      grafana_alert_datasource
    )}`
  }
  if (cluster_name) {
    url += `&var-cluster=${encodeURIComponent(cluster_name)}`
  }
  if (modelName) {
    url += `&var-model_name=${encodeURIComponent(modelName)}`
  }

  return url
}
