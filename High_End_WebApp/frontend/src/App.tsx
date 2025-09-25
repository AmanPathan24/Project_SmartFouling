import React, { useState, useEffect } from 'react'
import { Toaster } from 'react-hot-toast'
import Header from './components/Header'
import Navigation from './components/Navigation'
import UploadPanel from './components/UploadPanel'
import ResultsPanel from './components/ResultsPanel'
import AnalyticsPanel from './components/AnalyticsPanel'
import Simple3DView from './components/Simple3DView'
import MaintenancePanel from './components/MaintenancePanel'
import ImageClassificationPanel from './components/ImageClassificationPanel'
import AlertsPanel from './components/AlertsPanel'
import AlertNotification from './components/AlertNotification'
import LoadingModal from './components/LoadingModal'
import { SessionProvider } from './contexts/SessionContext'
import { ApiProvider } from './contexts/ApiContext'
import { AnalyticsProvider } from './contexts/AnalyticsContext'

type TabType = 'upload' | 'results' | 'analytics' | '3d-viz' | 'maintenance' | 'classification' | 'alerts'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('upload')
  const [isLoading, setIsLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [loadingText, setLoadingText] = useState('Initializing...')

  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab)
  }

  const startLoading = (text: string, progress: number = 0) => {
    setLoadingText(text)
    setLoadingProgress(progress)
    setIsLoading(true)
  }

  const stopLoading = () => {
    setIsLoading(false)
    setLoadingProgress(0)
  }

  const updateLoadingProgress = (progress: number, text: string) => {
    setLoadingProgress(progress)
    setLoadingText(text)
  }

  return (
    <ApiProvider>
      <SessionProvider>
        <AnalyticsProvider>
          <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
            {/* Floating Background Shapes */}
            <div className="floating-shapes">
              <div className="shape"></div>
              <div className="shape"></div>
              <div className="shape"></div>
            </div>

            {/* Header */}
            <Header />

            {/* Navigation */}
            <Navigation 
              activeTab={activeTab} 
              onTabChange={handleTabChange}
            />

            {/* Main Content */}
            <main className="max-w-7xl mx-auto p-6 relative z-10">
              {activeTab === 'upload' && (
                <UploadPanel 
                  onStartLoading={startLoading}
                  onStopLoading={stopLoading}
                  onUpdateProgress={updateLoadingProgress}
                />
              )}
              
              {activeTab === 'results' && (
                <ResultsPanel 
                  onStartLoading={startLoading}
                  onStopLoading={stopLoading}
                />
              )}
              
              {activeTab === 'analytics' && (
                <AnalyticsPanel />
              )}
              
              {activeTab === '3d-viz' && (
                <Simple3DView />
              )}
              
              {activeTab === 'maintenance' && (
                <MaintenancePanel />
              )}
              
              {activeTab === 'classification' && (
                <ImageClassificationPanel />
              )}
              
              {activeTab === 'alerts' && (
                <AlertsPanel />
              )}
            </main>

            {/* Loading Modal */}
            <LoadingModal 
              isOpen={isLoading}
              progress={loadingProgress}
              text={loadingText}
            />

            {/* Toast Notifications */}
            <Toaster 
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: 'rgba(30, 41, 59, 0.9)',
                  color: 'white',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  backdropFilter: 'blur(10px)',
                },
              }}
            />

            {/* Invasive Species Alert Notifications */}
            <AlertNotification />
          </div>
        </AnalyticsProvider>
      </SessionProvider>
    </ApiProvider>
  )
}

export default App
