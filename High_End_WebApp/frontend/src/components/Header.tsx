import React, { useState, useEffect } from 'react'
import { Ship, Plus, Activity } from 'lucide-react'
import { useSession } from '../contexts/SessionContext'

const Header: React.FC = () => {
  const { sessions } = useSession()
  const [lastAnalysis, setLastAnalysis] = useState<string>('--')

  useEffect(() => {
    if (sessions.length > 0) {
      const lastSession = sessions.find(s => s.status === 'completed')
      if (lastSession?.completed_at) {
        const date = new Date(lastSession.completed_at)
        setLastAnalysis(date.toLocaleDateString())
      }
    }
  }, [sessions])

  return (
    <header className="glass sticky top-0 z-40 p-6">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full blur-lg opacity-70"></div>
            <Ship className="text-white text-3xl relative z-10" />
          </div>
          <div>
            <h1 className="text-3xl font-bold gradient-text">Marine Biofouling</h1>
            <p className="text-sm text-gray-300 font-light">AI-Powered Detection System</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-6">
          <div className="hidden md:flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-gray-300">System Status:</span>
              <span className="text-green-400 font-medium">Online</span>
            </div>
            <div className="hidden lg:block text-right">
              <p className="text-xs text-gray-400 uppercase tracking-wide">Last Analysis</p>
              <span className="text-sm font-medium">{lastAnalysis}</span>
            </div>
          </div>
          
          <button 
            className="glass-button px-6 py-3 font-medium flex items-center space-x-2"
            onClick={() => window.location.reload()}
          >
            <Plus className="w-4 h-4" />
            <span>New Session</span>
          </button>
        </div>
      </div>
    </header>
  )
}

export default Header
