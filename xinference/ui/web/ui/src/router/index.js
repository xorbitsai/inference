import { useContext, useEffect, useState } from 'react'
import { Navigate, useRoutes } from 'react-router-dom'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../components/apiContext'
import Layout from '../scenes/_layout'
import ApiKeyManagement from '../scenes/apikey_management'
import AuditLog from '../scenes/audit_log'
import ChangePassword from '../scenes/change_password'
import ClusterInfo from '../scenes/cluster_info'
import LaunchModel from '../scenes/launch_model'
import Login from '../scenes/login/login'
import Logs from '../scenes/logs'
import Monitoring from '../scenes/monitoring'
import RegisterModel from '../scenes/register_model'
import RunningModels from '../scenes/running_models'
import SecuritySettings from '../scenes/security_settings'
import Setup from '../scenes/setup/setup'
import UserManagement from '../scenes/user_management'
import { hasPermission, parseTokenFromSession } from '../utils/jwt'

const LoginAuth = () => {
  const [authority, setAuthority] = useState(true)
  // null = not checked yet, so we don't flash the login form before we
  // know whether this instance still needs its first admin account.
  const [needsSetup, setNeedsSetup] = useState(null)

  const navigate = useNavigate()
  const { endPoint } = useContext(ApiContext)

  useEffect(() => {
    fetch(endPoint + '/v1/cluster/auth', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }).then((res) => {
      if (res.ok) {
        res.json().then((data) => {
          setAuthority(data.auth)
          sessionStorage.setItem('auth', String(data.auth)) // sessionStorage only can set string value
        })
      }
    })
  }, [])

  useEffect(() => {
    if (!authority) return
    fetch(endPoint + '/v1/admin/setup/status', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })
      .then((res) => (res.ok ? res.json() : { needs_setup: false }))
      .then((data) => setNeedsSetup(Boolean(data.needs_setup)))
      .catch(() => setNeedsSetup(false))
  }, [authority])

  useEffect(() => {
    if (!authority) navigate('/launch_model/llm')
  }, [authority])

  if (!authority) return null
  if (needsSetup === null) return null
  if (needsSetup) return <Setup />
  return <Login />
}

const routes = [
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        path: '/',
        element: <Navigate to="launch_model/llm" replace />,
      },
      {
        path: 'launch_model/:Modeltype/:subType?',
        element: <LaunchModel />,
      },
      {
        path: 'running_models/:runningModelType',
        element: <RunningModels />,
      },
      {
        path: 'register_model/:registerModelType/:model_name?',
        element: <RegisterModel />,
      },
      {
        path: 'cluster_info',
        element: (
          <PermissionGate requiredScope="admin">
            <ClusterInfo />
          </PermissionGate>
        ),
      },
      {
        path: 'monitoring',
        element: (
          <PermissionGate requiredScope="monitor:view">
            <Monitoring />
          </PermissionGate>
        ),
      },
      {
        path: 'logs',
        element: (
          <PermissionGate requiredScope="logs:list">
            <Logs />
          </PermissionGate>
        ),
      },
      {
        path: 'user_management',
        element: (
          <PermissionGate requiredScope="users:manage">
            <UserManagement />
          </PermissionGate>
        ),
      },
      {
        path: 'apikey_management',
        element: (
          <PermissionGate
            requiredScope="keys:create"
            alternateScope="keys:manage"
          >
            <ApiKeyManagement />
          </PermissionGate>
        ),
      },
      {
        path: 'security_settings',
        element: (
          <PermissionGate requiredScope="admin">
            <SecuritySettings />
          </PermissionGate>
        ),
      },
      {
        path: 'audit_log',
        element: (
          <PermissionGate requiredScope="admin">
            <AuditLog />
          </PermissionGate>
        ),
      },
    ],
  },
  {
    path: '/login',
    element: <LoginAuth />,
  },
  {
    path: '/change_password',
    element: <ChangePassword />,
  },
  {
    path: '*',
    element: <Navigate to="launch_model/llm" replace />,
  },
]
const WraperRoutes = () => {
  let element = useRoutes(routes)
  return element
}

// `PermissionGate` redirects to `/` when the current session lacks the
// required scope. `admin` is a wildcard (passes any gate). Routes that
// are A-class (launch / running / register model) are intentionally NOT
// wrapped — they remain visible to all authenticated users and rely on
// backend 403 as the only enforcement, matching the menu-visibility
// contract in MenuSide.js.
function PermissionGate({ requiredScope, alternateScope, children }) {
  const { scopes } = parseTokenFromSession() || {}
  const ok =
    !requiredScope ||
    hasPermission(scopes, requiredScope) ||
    (alternateScope && hasPermission(scopes, alternateScope))
  return ok ? children : <Navigate to="/" replace />
}

export default WraperRoutes
