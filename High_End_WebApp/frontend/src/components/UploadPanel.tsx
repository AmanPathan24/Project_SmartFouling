import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Brain, FileImage } from 'lucide-react'
import { useSession } from '../contexts/SessionContext'
import { useApi } from '../contexts/ApiContext'
import toast from 'react-hot-toast'

interface UploadPanelProps {
  onStartLoading: (text: string, progress?: number) => void
  onStopLoading: () => void
  onUpdateProgress: (progress: number, text: string) => void
}

const UploadPanel: React.FC<UploadPanelProps> = ({
  onStartLoading,
  onStopLoading,
  onUpdateProgress,
}) => {
  const { currentSession, createSession, uploadImages, preprocessImages, analyzeImages, sessions, getAllSessions, getSession, setCurrentSession } = useSession()
  const { api } = useApi()
  
  const [sessionName, setSessionName] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])


  const onDrop = useCallback((acceptedFiles: File[]) => {
    const imageFiles = acceptedFiles.filter(file => file.type.startsWith('image/'))
    setUploadedFiles(prev => [...prev, ...imageFiles])
    toast.success(`Added ${imageFiles.length} images`)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.webp']
    },
    multiple: true
  })

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleCreateSession = async () => {
    if (!sessionName.trim()) {
      toast.error('Please enter a session name')
      return
    }

    try {
      const sessionId = await createSession(sessionName)
      toast.success('Session created successfully!')
    } catch (error) {
      console.error('Failed to create session:', error)
      toast.error('Failed to create session')
    }
  }

  const handleUploadImages = async () => {
    if (!currentSession) {
      toast.error('Please create a session first')
      return
    }

    if (uploadedFiles.length === 0) {
      toast.error('Please select images to upload')
      return
    }

    try {
      onStartLoading('Processing images...', 10)
      
      // Use the new combined upload and process endpoint
      const formData = new FormData()
      uploadedFiles.forEach(file => {
        formData.append('files', file)
      })

      onUpdateProgress(30, 'Uploading images...')
      const response = await api.post(`/sessions/${currentSession.session_id}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      onUpdateProgress(60, 'Running AI analysis...')
      
      // Update current session with results
      const updatedSession = await getSession(currentSession.session_id)
      if (updatedSession) {
        setCurrentSession(updatedSession)
      }

      onUpdateProgress(100, 'Analysis completed!')
      setUploadedFiles([])
      onStopLoading()
      toast.success('Images processed successfully!')
    } catch (error) {
      console.error('Failed to process images:', error)
      onStopLoading()
      toast.error('Failed to process images')
    }
  }

  const handleCreateSessionAndUpload = async () => {
    if (!sessionName.trim()) {
      toast.error('Please enter a session name')
      return
    }

    if (uploadedFiles.length === 0) {
      toast.error('Please select images to upload')
      return
    }

    try {
      onStartLoading('Creating session and processing images...', 5)
      
      // Create session first
      console.log('Creating session...')
      const sessionId = await createSession(sessionName)
      console.log('Session created with ID:', sessionId)
      toast.success('Session created successfully!')
      
      onUpdateProgress(20, 'Uploading images...')
      
      // Then upload and process images
      const formData = new FormData()
      uploadedFiles.forEach(file => {
        formData.append('files', file)
      })

      console.log('Uploading images to session:', sessionId)
      const response = await api.post(`/sessions/${sessionId}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      console.log('Upload response:', response.data)

      onUpdateProgress(80, 'Running AI analysis...')
      
      // Update current session with results
      console.log('Getting updated session data...')
      const updatedSession = await getSession(sessionId)
      console.log('Updated session:', updatedSession)
      if (updatedSession) {
        setCurrentSession(updatedSession)
      }

      onUpdateProgress(100, 'Analysis completed!')
      setUploadedFiles([])
      setSessionName('')
      onStopLoading()
      toast.success('Session created and images processed successfully!')
    } catch (error) {
      console.error('Failed to create session and process images:', error)
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        statusText: error.response?.statusText
      })
      onStopLoading()
      toast.error(`Failed to create session and process images: ${error.message}`)
    }
  }


  return (
    <div className="max-w-4xl mx-auto">
      {/* Upload Section */}
      <div className="glass-card p-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold gradient-text flex items-center">
            <Upload className="w-6 h-6 mr-3" />
            Upload Images
          </h2>
          <span className="px-3 py-1 text-xs font-medium bg-green-500/20 text-green-300 rounded-full border border-green-500/30">
            AI Ready
          </span>
        </div>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-3">Session Name</label>
            <input 
              type="text" 
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50 transition-all" 
              placeholder="e.g., Hull Inspection - Port Side"
            />
          </div>
          
          
          {/* Primary Action Button */}
          <button
            onClick={handleCreateSessionAndUpload}
            disabled={!sessionName.trim() || uploadedFiles.length === 0}
            className="w-full glass-button px-8 py-4 font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 mb-4"
          >
            <Brain className="w-6 h-6" />
            <span>Create Session & Start AI Analysis</span>
          </button>

          {/* Secondary Actions */}
          <div className="grid md:grid-cols-2 gap-4">
            <button
              onClick={handleCreateSession}
              disabled={!sessionName.trim()}
              className="glass-button px-6 py-3 font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              <Upload className="w-4 h-4" />
              <span>Create Session Only</span>
            </button>
            
            <button
              onClick={handleUploadImages}
              disabled={!currentSession || uploadedFiles.length === 0}
              className="glass-button px-6 py-3 font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              <Brain className="w-4 h-4" />
              <span>Upload to Existing Session</span>
            </button>
          </div>
          
          {/* File Drop Zone */}
          <div 
            {...getRootProps()} 
            className={`drop-zone ${isDragActive ? 'dragover' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full blur-2xl opacity-30"></div>
              <FileImage className="w-16 h-16 text-white relative z-10 mx-auto" />
            </div>
            <h3 className="text-xl font-semibold mb-2">
              {isDragActive ? 'Drop images here' : 'Drop images here'}
            </h3>
            <p className="text-gray-400 mb-4">or click to browse your files</p>
            <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
              <span className="flex items-center"><span className="w-2 h-2 bg-green-400 rounded-full mr-1"></span>JPG</span>
              <span className="flex items-center"><span className="w-2 h-2 bg-green-400 rounded-full mr-1"></span>PNG</span>
              <span className="flex items-center"><span className="w-2 h-2 bg-green-400 rounded-full mr-1"></span>Multiple files</span>
            </div>
          </div>
          
          {/* File List */}
          {uploadedFiles.length > 0 && (
            <div className="space-y-2">
              <h4 className="font-semibold text-gray-300">Selected Files:</h4>
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between glass p-3 rounded-lg">
                  <span className="text-sm text-gray-300">{file.name}</span>
                  <button
                    onClick={() => removeFile(index)}
                    className="text-red-400 hover:text-red-300 text-sm"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default UploadPanel
