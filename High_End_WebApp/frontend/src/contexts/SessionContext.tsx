import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { useApi } from './ApiContext'
import toast from 'react-hot-toast'

export interface ImageMetadata {
  filename: string
  original_path: string
  preprocessed_path?: string
  segmentation_path?: string
  source: string
  uploaded_at: string
  size: number
  dimensions: [number, number]
}

export interface Detection {
  species: string
  scientific_name: string
  confidence: number
  coverage_percentage: number
  bbox: {
    x: number
    y: number
    width: number
    height: number
  }
}

export interface AnalysisResult {
  image_id: string
  detections: Detection[]
  total_coverage: number
  dominant_species: string
  processing_time: number
}

export interface SessionAnalytics {
  total_coverage: number
  species_count: number
  dominant_species: string
  avg_confidence: number
  total_detections: number
  fuel_cost_impact: number
  maintenance_cost: number
  cleaning_urgency: 'low' | 'medium' | 'high' | 'critical'
  species_distribution: Record<string, number>
}

export interface Session {
  session_id: string
  session_name: string
  status: 'created' | 'processing' | 'completed' | 'failed'
  created_at: string
  updated_at?: string
  completed_at?: string
  images: ImageMetadata[]
  analysis_results?: AnalysisResult[]
  analytics?: SessionAnalytics
}

interface SessionContextType {
  currentSession: Session | null
  sessions: Session[]
  isLoading: boolean
  createSession: (sessionName: string) => Promise<string>
  uploadImages: (sessionId: string, files: File[], source: string) => Promise<void>
  preprocessImages: (sessionId: string) => Promise<void>
  analyzeImages: (sessionId: string) => Promise<void>
  getSession: (sessionId: string) => Promise<Session | null>
  getAllSessions: () => Promise<void>
  setCurrentSession: (session: Session | null) => void
}

const SessionContext = createContext<SessionContextType | undefined>(undefined)

export const useSession = () => {
  const context = useContext(SessionContext)
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider')
  }
  return context
}

interface SessionProviderProps {
  children: ReactNode
}

export const SessionProvider: React.FC<SessionProviderProps> = ({ children }) => {
  const [currentSession, setCurrentSession] = useState<Session | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const { api } = useApi()

  const createSession = useCallback(async (
    sessionName: string
  ): Promise<string> => {
    setIsLoading(true)
    try {
      const requestData = {
        session_name: sessionName
      }

      const response = await api.post('/sessions', requestData)

      const { session_id } = response.data
      
      // Create session object
      const newSession: Session = {
        session_id,
        session_name: sessionName,
        status: 'created',
        created_at: new Date().toISOString(),
        images: [],
      }

      setCurrentSession(newSession)
      setSessions(prev => [newSession, ...prev])
      
      toast.success('Session created successfully!')
      return session_id
    } catch (error) {
      console.error('Failed to create session:', error)
      toast.error('Failed to create session')
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [api])

  const uploadImages = useCallback(async (
    sessionId: string,
    files: File[],
    source: string = 'manual'
  ): Promise<void> => {
    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('source', source)
      
      files.forEach(file => {
        formData.append('files', file)
      })

      const response = await api.post(`/sessions/${sessionId}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      // Update current session with uploaded images
      if (currentSession?.session_id === sessionId) {
        const updatedSession = await getSession(sessionId)
        if (updatedSession) {
          setCurrentSession(updatedSession)
        }
      }

      toast.success(`Successfully uploaded ${files.length} images!`)
    } catch (error) {
      console.error('Failed to upload images:', error)
      toast.error('Failed to upload images')
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [api, currentSession])

  const preprocessImages = useCallback(async (sessionId: string): Promise<void> => {
    setIsLoading(true)
    try {
      await api.post(`/sessions/${sessionId}/preprocess`)
      
      // Update current session
      if (currentSession?.session_id === sessionId) {
        const updatedSession = await getSession(sessionId)
        if (updatedSession) {
          setCurrentSession(updatedSession)
        }
      }

      toast.success('Images preprocessed successfully!')
    } catch (error) {
      console.error('Failed to preprocess images:', error)
      toast.error('Failed to preprocess images')
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [api, currentSession])

  const analyzeImages = useCallback(async (sessionId: string): Promise<void> => {
    setIsLoading(true)
    try {
      const response = await api.post(`/sessions/${sessionId}/analyze`)
      
      // Update current session
      const updatedSession = await getSession(sessionId)
      if (updatedSession) {
        setCurrentSession(updatedSession)
      }

      toast.success('Analysis completed successfully!')
    } catch (error) {
      console.error('Failed to analyze images:', error)
      toast.error('Failed to analyze images')
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [api])

  const getSession = useCallback(async (sessionId: string): Promise<Session | null> => {
    try {
      const response = await api.get(`/sessions/${sessionId}`)
      return response.data as Session
    } catch (error) {
      console.error('Failed to get session:', error)
      return null
    }
  }, [api])

  const getAllSessions = useCallback(async (): Promise<void> => {
    try {
      const response = await api.get('/sessions')
      setSessions(response.data.sessions || [])
    } catch (error) {
      console.error('Failed to get sessions:', error)
      toast.error('Failed to load sessions')
    }
  }, [api])

  // Load sessions on mount
  React.useEffect(() => {
    getAllSessions()
  }, [getAllSessions])

  const value: SessionContextType = {
    currentSession,
    sessions,
    isLoading,
    createSession,
    uploadImages,
    preprocessImages,
    analyzeImages,
    getSession,
    getAllSessions,
    setCurrentSession,
  }

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  )
}
