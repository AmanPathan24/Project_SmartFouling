import React, { createContext, useContext, useState, ReactNode } from 'react'
import axios, { AxiosInstance } from 'axios'

interface ApiContextType {
  api: AxiosInstance
  isConnected: boolean
  checkConnection: () => Promise<boolean>
}

const ApiContext = createContext<ApiContextType | undefined>(undefined)

export const useApi = () => {
  const context = useContext(ApiContext)
  if (context === undefined) {
    throw new Error('useApi must be used within an ApiProvider')
  }
  return context
}

interface ApiProviderProps {
  children: ReactNode
}

export const ApiProvider: React.FC<ApiProviderProps> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false)

  const api = axios.create({
    baseURL: '/api',
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  })

  // Add request interceptor
  api.interceptors.request.use(
    (config) => {
      console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`)
      return config
    },
    (error) => {
      console.error('Request error:', error)
      return Promise.reject(error)
    }
  )

  // Add response interceptor
  api.interceptors.response.use(
    (response) => {
      return response
    },
    (error) => {
      console.error('Response error:', error.response?.data || error.message)
      return Promise.reject(error)
    }
  )

  const checkConnection = async (): Promise<boolean> => {
    try {
      const response = await api.get('/health')
      setIsConnected(response.status === 200)
      return response.status === 200
    } catch (error) {
      console.error('Connection check failed:', error)
      setIsConnected(false)
      return false
    }
  }

  const value: ApiContextType = {
    api,
    isConnected,
    checkConnection,
  }

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  )
}
