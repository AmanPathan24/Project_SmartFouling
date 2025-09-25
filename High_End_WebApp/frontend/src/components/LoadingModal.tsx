import React from 'react'
import { Loader2, Brain, Microscope, Zap } from 'lucide-react'

interface LoadingModalProps {
  isOpen: boolean
  progress: number
  text: string
}

const LoadingModal: React.FC<LoadingModalProps> = ({ isOpen, progress, text }) => {
  if (!isOpen) return null

  const getIcon = () => {
    if (text.toLowerCase().includes('upload')) return <Zap className="w-8 h-8" />
    if (text.toLowerCase().includes('preprocess')) return <Microscope className="w-8 h-8" />
    if (text.toLowerCase().includes('analysis') || text.toLowerCase().includes('ai')) return <Brain className="w-8 h-8" />
    return <Loader2 className="w-8 h-8 animate-spin" />
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-md flex items-center justify-center z-50">
      <div className="glass-card p-12 text-center max-w-md mx-4">
        <div className="relative mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full blur-2xl opacity-50 animate-pulse"></div>
          <div className="relative z-10 flex justify-center">
            {getIcon()}
          </div>
        </div>
        
        <h3 className="text-2xl font-bold gradient-text mb-3">AI Processing</h3>
        <p className="text-gray-400 text-lg mb-6">{text}</p>
        
        <div className="progress-bar mb-4">
          <div 
            className="progress-fill transition-all duration-500 ease-out"
            style={{ width: `${Math.min(progress, 100)}%` }}
          ></div>
        </div>
        
        <p className="text-sm text-gray-500">
          {progress > 0 ? `${Math.round(progress)}% Complete` : 'Initializing...'}
        </p>
        
        {/* Animated dots */}
        <div className="flex justify-center space-x-1 mt-4">
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  )
}

export default LoadingModal
