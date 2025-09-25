import React, { useState, useEffect } from 'react'
import { Microscope, Image, Eye, Download, AlertTriangle, CheckCircle, FileText } from 'lucide-react'
import { useSession, Session } from '../contexts/SessionContext'
import { useApi } from '../contexts/ApiContext'
import toast from 'react-hot-toast'

interface ResultsPanelProps {
  onStartLoading: (text: string, progress?: number) => void
  onStopLoading: () => void
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ onStartLoading, onStopLoading }) => {
  const { sessions, currentSession, setCurrentSession } = useSession()
  const { api } = useApi()
  const [selectedSession, setSelectedSession] = useState<Session | null>(currentSession)
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)

  useEffect(() => {
    if (currentSession) {
      setSelectedSession(currentSession)
    }
  }, [currentSession])

  const handleSessionSelect = (session: Session) => {
    setSelectedSession(session)
    setCurrentSession(session)
    setSelectedImageIndex(0)
  }

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical': return 'text-red-400 bg-red-500/20 border-red-500/30'
      case 'high': return 'text-orange-400 bg-orange-500/20 border-orange-500/30'
      case 'medium': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      case 'low': return 'text-green-400 bg-green-500/20 border-green-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  const getUrgencyIcon = (urgency: string) => {
    switch (urgency) {
      case 'critical': return <AlertTriangle className="w-4 h-4" />
      case 'high': return <AlertTriangle className="w-4 h-4" />
      case 'medium': return <AlertTriangle className="w-4 h-4" />
      case 'low': return <CheckCircle className="w-4 h-4" />
      default: return <CheckCircle className="w-4 h-4" />
    }
  }

  const handleGenerateReport = async () => {
    if (!selectedSession) {
      toast.error('No session selected')
      return
    }

    try {
      onStartLoading('Generating LLM-powered report...', 50)
      
      const response = await api.post(`/sessions/${selectedSession.session_id}/generate-report`, {
        session_id: selectedSession.session_id,
        analysis_data: selectedSession
      })

      onStopLoading()
      toast.success('Report generated successfully!')
      
      // Download the report
      if (response.data.report_path) {
        const link = document.createElement('a')
        // Extract just the filename from the full path
        const filename = response.data.report_path.split('/').pop()
        link.href = `/api/images/outputs/${filename}`
        link.download = `${selectedSession.session_name}_report.pdf`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      }
    } catch (error) {
      console.error('Failed to generate report:', error)
      onStopLoading()
      toast.error('Failed to generate report')
    }
  }

  if (!selectedSession) {
    return (
      <div className="glass-card p-12 text-center">
        <div className="relative mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-blue-500 rounded-full blur-2xl opacity-30"></div>
          <Microscope className="w-16 h-16 text-white relative z-10 mx-auto" />
        </div>
        <h3 className="text-2xl font-bold gradient-text mb-4">Select a Session</h3>
        <p className="text-gray-400 text-lg">Choose a completed session to view detailed results and insights</p>
        
        <div className="mt-8 grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sessions.filter(s => s.status === 'completed').map((session) => (
            <div 
              key={session.session_id}
              onClick={() => handleSessionSelect(session)}
              className="glass-card p-6 cursor-pointer hover:scale-105 transition-all duration-300"
            >
              <h4 className="font-semibold text-white mb-2">{session.session_name}</h4>
              <p className="text-sm text-gray-400 mb-3">
                {new Date(session.created_at).toLocaleDateString()}
              </p>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300">
                  {session.images?.length || 0} images
                </span>
                <span className="px-2 py-1 text-xs bg-green-500/20 text-green-300 rounded-full">
                  completed
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const currentImage = selectedSession.images?.[selectedImageIndex]

  return (
    <div className="space-y-6">
      {/* Session Header */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold gradient-text">{selectedSession.session_name}</h2>
            <p className="text-gray-400">
              Created: {new Date(selectedSession.created_at).toLocaleString()}
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <span className="px-3 py-1 text-xs font-medium bg-green-500/20 text-green-300 rounded-full border border-green-500/30">
              Completed
            </span>
            <button 
              onClick={handleGenerateReport}
              className="glass-button px-4 py-2 text-sm flex items-center space-x-2"
            >
              <FileText className="w-4 h-4" />
              <span>Generate LLM Report</span>
            </button>
            <button className="glass-button px-4 py-2 text-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export Data</span>
            </button>
          </div>
        </div>

        {/* Analytics Summary */}
        {selectedSession.analytics && (
          <div className="grid md:grid-cols-4 gap-4">
            <div className="glass p-4 rounded-xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Total Coverage</span>
                <span className="text-2xl font-bold text-blue-400">
                  {selectedSession.analytics.total_coverage.toFixed(1)}%
                </span>
              </div>
            </div>
            
            <div className="glass p-4 rounded-xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Species Count</span>
                <span className="text-2xl font-bold text-purple-400">
                  {selectedSession.analytics.species_count}
                </span>
              </div>
            </div>
            
            <div className="glass p-4 rounded-xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Dominant Species</span>
                <span className="text-lg font-bold text-green-400">
                  {selectedSession.analytics.dominant_species}
                </span>
              </div>
            </div>
            
            <div className="glass p-4 rounded-xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Cleaning Urgency</span>
                <span className={`px-2 py-1 text-xs rounded-full border flex items-center space-x-1 ${getUrgencyColor(selectedSession.analytics.cleaning_urgency)}`}>
                  {getUrgencyIcon(selectedSession.analytics.cleaning_urgency)}
                  <span>{selectedSession.analytics.cleaning_urgency}</span>
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Image Results */}
      {selectedSession.images && selectedSession.images.length > 0 && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Image Navigation */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold gradient-text mb-4 flex items-center">
              <Image className="w-5 h-5 mr-2" />
              Images ({selectedSession.images.length})
            </h3>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {selectedSession.images.map((image, index) => (
                <div 
                  key={index}
                  onClick={() => setSelectedImageIndex(index)}
                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                    index === selectedImageIndex 
                      ? 'glass bg-blue-500/20 border border-blue-500/30' 
                      : 'glass hover:bg-white/10'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-white">{image.filename}</span>
                    <div className="flex items-center space-x-2">
                      {image.preprocessed_path && (
                        <span className="w-2 h-2 bg-green-400 rounded-full" title="Preprocessed"></span>
                      )}
                      {image.segmentation_path && (
                        <span className="w-2 h-2 bg-blue-400 rounded-full" title="Segmented"></span>
                      )}
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    {image.dimensions[0]} × {image.dimensions[1]}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Image Display */}
          <div className="lg:col-span-2 space-y-6">
            {currentImage && (
              <>
                {/* Original Image */}
                <div className="glass-card p-6">
                  <h3 className="text-lg font-bold gradient-text mb-4 flex items-center">
                    <Eye className="w-5 h-5 mr-2" />
                    Original Image
                  </h3>
                  <div className="relative">
                    <img 
                      src={`/api/images/original/${currentImage.original_path.split('/').pop()}`} 
                      alt={currentImage.filename}
                      className="w-full h-64 object-cover rounded-xl"
                      onError={(e) => {
                        // Fallback to a placeholder if image not found
                        e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjI0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzIwIiBoZWlnaHQ9IjI0MCIgZmlsbD0iIzMzMzMzMyIvPjx0ZXh0IHg9IjE2MCIgeT0iMTIwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2NjY2NjYiIHRleHQtYW5jaG9yPSJtaWRkbGUiPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4='
                      }}
                    />
                    <div className="absolute top-4 right-4 glass px-3 py-1 rounded-full text-sm">
                      Original
                    </div>
                  </div>
                </div>

                {/* Preprocessed Image */}
                {currentImage.preprocessed_path && (
                  <div className="glass-card p-6">
                    <h3 className="text-lg font-bold gradient-text mb-4 flex items-center">
                      <Microscope className="w-5 h-5 mr-2" />
                      Preprocessed Image
                    </h3>
                    <div className="relative">
                      <img 
                        src={`/api/images/preprocessed/${currentImage.preprocessed_path.split('/').pop()}`}
                        alt={`Preprocessed ${currentImage.filename}`}
                        className="w-full h-64 object-cover rounded-xl"
                        onError={(e) => {
                          e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjI0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzIwIiBoZWlnaHQ9IjI0MCIgZmlsbD0iIzMzMzMzMyIvPjx0ZXh0IHg9IjE2MCIgeT0iMTIwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2NjY2NjYiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlByZXByb2Nlc3NlZCBJbWFnZTwvdGV4dD48L3N2Zz4='
                        }}
                      />
                      <div className="absolute top-4 right-4 glass px-3 py-1 rounded-full text-sm bg-green-500/20 text-green-300">
                        Enhanced
                      </div>
                    </div>
                  </div>
                )}

                {/* Segmentation Mask */}
                {currentImage.segmentation_path && (
                  <div className="glass-card p-6">
                    <h3 className="text-lg font-bold gradient-text mb-4 flex items-center">
                      <Microscope className="w-5 h-5 mr-2" />
                      Segmentation Mask
                    </h3>
                    <div className="relative">
                      <img 
                        src={`/api/images/segmented/${currentImage.segmentation_path.split('/').pop()}`}
                        alt={`Segmentation ${currentImage.filename}`}
                        className="w-full h-64 object-cover rounded-xl"
                        onError={(e) => {
                          e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjI0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzIwIiBoZWlnaHQ9IjI0MCIgZmlsbD0iIzMzMzMzMyIvPjx0ZXh0IHg9IjE2MCIgeT0iMTIwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2NjY2NjYiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlNlZ21lbnRhdGlvbiBNYXNrPC90ZXh0Pjwvc3ZnPg=='
                        }}
                      />
                      <div className="absolute top-4 right-4 glass px-3 py-1 rounded-full text-sm bg-blue-500/20 text-blue-300">
                        AI Detected
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Detection Results */}
      {selectedSession.analysis_results && selectedSession.analysis_results.length > 0 && (
        <div className="glass-card p-6">
          <h3 className="text-lg font-bold gradient-text mb-4">Detection Results</h3>
          
          {selectedSession.analysis_results[selectedImageIndex] && (
            <div className="space-y-4">
              {selectedSession.analysis_results[selectedImageIndex].detections.map((detection, index) => (
                <div key={index} className="glass p-4 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-white">{detection.species}</h4>
                    <span className="text-sm text-gray-400">
                      Confidence: {(detection.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-300 mb-2">
                    <strong>Scientific Name:</strong> {detection.scientific_name}
                  </p>
                  
                  <div className="grid md:grid-cols-3 gap-4">
                    <div>
                      <span className="text-sm text-gray-400">Coverage:</span>
                      <span className="ml-2 font-semibold text-blue-400">
                        {detection.coverage_percentage.toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-sm text-gray-400">Position:</span>
                      <span className="ml-2 font-semibold text-green-400">
                        ({detection.bbox.x}, {detection.bbox.y})
                      </span>
                    </div>
                    <div>
                      <span className="text-sm text-gray-400">Size:</span>
                      <span className="ml-2 font-semibold text-purple-400">
                        {detection.bbox.width} × {detection.bbox.height}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ResultsPanel
