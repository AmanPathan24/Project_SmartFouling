import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { useApi } from './ApiContext'

export interface ChartData {
  time_series: Array<{
    date: string
    fouling_density: number
    fuel_cost: number
    maintenance_cost: number
  }>
  species_distribution: Array<{
    species: string
    coverage: number
    count: number
  }>
  cost_projection: Array<{
    delay_days: number
    cleaning_cost: number
    fuel_cost: number
    total_cost: number
  }>
  summary: {
    total_sessions: number
    avg_coverage: number
    dominant_species: string
    total_cost_saved: number
  }
}

export interface MaintenanceTask {
  id: number
  title: string
  description: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  estimated_cost: number
  estimated_duration: string
  recommended_date: string
  species: string[]
  coverage: number
  status?: 'pending' | 'in_progress' | 'completed'
  created_at?: string
}

interface AnalyticsContextType {
  chartData: ChartData | null
  maintenanceTasks: MaintenanceTask[]
  isLoading: boolean
  fetchChartData: (sessionId?: string, timeRange?: string) => Promise<void>
  fetchMaintenanceTasks: () => Promise<void>
  addMaintenanceTask: (task: Omit<MaintenanceTask, 'id' | 'created_at'>) => Promise<void>
  updateMaintenanceTask: (taskId: number, updates: Partial<MaintenanceTask>) => Promise<void>
}

const AnalyticsContext = createContext<AnalyticsContextType | undefined>(undefined)

export const useAnalytics = () => {
  const context = useContext(AnalyticsContext)
  if (context === undefined) {
    throw new Error('useAnalytics must be used within an AnalyticsProvider')
  }
  return context
}

interface AnalyticsProviderProps {
  children: ReactNode
}

export const AnalyticsProvider: React.FC<AnalyticsProviderProps> = ({ children }) => {
  const [chartData, setChartData] = useState<ChartData | null>(null)
  const [maintenanceTasks, setMaintenanceTasks] = useState<MaintenanceTask[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const { api } = useApi()

  const fetchChartData = useCallback(async (
    sessionId?: string,
    timeRange: string = '30d'
  ): Promise<void> => {
    setIsLoading(true)
    try {
      const params = new URLSearchParams()
      if (sessionId) params.append('session_id', sessionId)
      params.append('time_range', timeRange)

      const response = await api.get(`/analytics/charts?${params}`)
      setChartData(response.data)
    } catch (error) {
      console.error('Failed to fetch chart data:', error)
    } finally {
      setIsLoading(false)
    }
  }, [api])

  const fetchMaintenanceTasks = useCallback(async (): Promise<void> => {
    setIsLoading(true)
    try {
      // For now, we'll use mock data since the backend endpoint doesn't exist yet
      const mockTasks: MaintenanceTask[] = [
        {
          id: 1,
          title: "Hull Cleaning - Port Side",
          description: "High barnacle density detected in port side mid-hull region",
          priority: "high",
          estimated_cost: 2500,
          estimated_duration: "4 hours",
          recommended_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          species: ["Barnacles", "Seaweed"],
          coverage: 45.2,
          status: "pending"
        },
        {
          id: 2,
          title: "Propeller Maintenance",
          description: "Moderate fouling on propeller blades affecting efficiency",
          priority: "medium",
          estimated_cost: 1800,
          estimated_duration: "2 hours",
          recommended_date: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          species: ["Mussels", "Seaweed"],
          coverage: 28.7,
          status: "pending"
        },
        {
          id: 3,
          title: "Routine Hull Inspection",
          description: "Quarterly routine inspection and minor cleaning",
          priority: "low",
          estimated_cost: 1200,
          estimated_duration: "6 hours",
          recommended_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          species: ["Sponges", "Anemones"],
          coverage: 12.1,
          status: "pending"
        }
      ]
      
      setMaintenanceTasks(mockTasks)
    } catch (error) {
      console.error('Failed to fetch maintenance tasks:', error)
    } finally {
      setIsLoading(false)
    }
  }, [api])

  const addMaintenanceTask = useCallback(async (
    task: Omit<MaintenanceTask, 'id' | 'created_at'>
  ): Promise<void> => {
    setIsLoading(true)
    try {
      // Mock implementation - in real app, this would call the backend
      const newTask: MaintenanceTask = {
        ...task,
        id: Date.now(), // Simple ID generation
        created_at: new Date().toISOString()
      }
      
      setMaintenanceTasks(prev => [...prev, newTask])
    } catch (error) {
      console.error('Failed to add maintenance task:', error)
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [])

  const updateMaintenanceTask = useCallback(async (
    taskId: number,
    updates: Partial<MaintenanceTask>
  ): Promise<void> => {
    setIsLoading(true)
    try {
      setMaintenanceTasks(prev => 
        prev.map(task => 
          task.id === taskId 
            ? { ...task, ...updates }
            : task
        )
      )
    } catch (error) {
      console.error('Failed to update maintenance task:', error)
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Load initial data
  React.useEffect(() => {
    fetchChartData()
    fetchMaintenanceTasks()
  }, [fetchChartData, fetchMaintenanceTasks])

  const value: AnalyticsContextType = {
    chartData,
    maintenanceTasks,
    isLoading,
    fetchChartData,
    fetchMaintenanceTasks,
    addMaintenanceTask,
    updateMaintenanceTask,
  }

  return (
    <AnalyticsContext.Provider value={value}>
      {children}
    </AnalyticsContext.Provider>
  )
}
