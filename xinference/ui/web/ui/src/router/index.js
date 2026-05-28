import { useContext, useEffect, useState } from 'react'
import { Navigate, useRoutes } from 'react-router-dom'
import { useNavigate } from 'react-router-dom'

import { ApiContext } from '../components/apiContext'
import Layout from '../scenes/_layout'
import ApiKeyManagement from '../scenes/apikey_management'
import ChangePassword from '../scenes/change_password'
import ClusterInfo from '../scenes/cluster_info'
import LaunchModel from '../scenes/launch_model'
import Login from '../scenes/login/login'
import Logs from '../scenes/logs'
import Monitoring from '../scenes/monitoring'
import RegisterModel from '../scenes/register_model'
import RunningModels from '../scenes/running_models'
import SecuritySettings from '../scenes/security_settings'
import UserManagement from '../scenes/user_management'

const LoginAuth = () => {
  const [authority, setAuthority] = useState(true)

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
    if (!authority) navigate('/launch_model/llm')
  }, [authority])

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
        element: <ClusterInfo />,
      },
      {
        path: 'monitoring',
        element: <Monitoring />,
      },
      {
        path: 'logs',
        element: <Logs />,
      },
      {
        path: 'user_management',
        element: <UserManagement />,
      },
      {
        path: 'apikey_management',
        element: <ApiKeyManagement />,
      },
      {
        path: 'security_settings',
        element: <SecuritySettings />,
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

export default WraperRoutes
