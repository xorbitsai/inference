/**
 * Build Grafana iframe URL for a specific sub-dashboard (kiosk mode).
 * @param {object} config - UI config from /v1/cluster/ui_config
 * @param {string} dashboardKey - sub-dashboard key (overview, model_load, llm_slo, gpu, host, security)
 * @param {string} theme - light/dark
 * @param {string} from - time range start
 * @param {string} to - time range end
 * @param {string} refresh - auto-refresh interval
 */
export const buildGrafanaUrl = (
  config,
  dashboardKey = 'overview',
  theme = 'light',
  from = 'now-1h',
  to = 'now',
  refresh
) => {
  const {
    grafana_url,
    grafana_dashboard_uid,
    grafana_dashboards,
    grafana_datasource,
    grafana_alert_datasource,
    cluster_name,
  } = config
  if (!grafana_url) return null

  // Get dashboard UID from grafana_dashboards map, fallback to legacy field
  const dashboardUid =
    grafana_dashboards?.[dashboardKey] ||
    grafana_dashboard_uid ||
    'xinference-overview'

  let url = `${grafana_url}/d/${dashboardUid}?orgId=1&kiosk&theme=${theme}&from=${encodeURIComponent(
    from
  )}&to=${encodeURIComponent(to)}`

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
  if (refresh !== undefined) {
    url += `&refresh=${encodeURIComponent(refresh)}`
  }

  return url
}

/**
 * Build Grafana single-panel embed URL (for model detail pages).
 * @param {object} config - UI config
 * @param {string} dashboardKey - sub-dashboard key
 * @param {number} panelId - panel ID
 * @param {string} modelName - model name filter
 * @param {string} theme - light/dark
 * @param {string} from - time range start
 * @param {string} to - time range end
 */
export const buildGrafanaPanelUrl = (
  config,
  dashboardKey = 'overview',
  panelId,
  modelName,
  theme = 'light',
  from = 'now-1h',
  to = 'now'
) => {
  const {
    grafana_url,
    grafana_dashboard_uid,
    grafana_dashboards,
    grafana_datasource,
    grafana_alert_datasource,
    cluster_name,
  } = config
  if (!grafana_url) return null

  const dashboardUid =
    grafana_dashboards?.[dashboardKey] ||
    grafana_dashboard_uid ||
    'xinference-overview'

  let url = `${grafana_url}/d-solo/${dashboardUid}?orgId=1&kiosk&theme=${theme}&panelId=${panelId}&from=${from}&to=${to}`

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
