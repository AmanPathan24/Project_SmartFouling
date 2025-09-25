import React from 'react'
import { Upload, Microscope, BarChart3, Box, Wrench, Brain, AlertTriangle } from 'lucide-react'

interface NavigationProps {
  activeTab: 'upload' | 'results' | 'analytics' | '3d-viz' | 'maintenance' | 'classification' | 'alerts'
  onTabChange: (tab: 'upload' | 'results' | 'analytics' | '3d-viz' | 'maintenance' | 'classification' | 'alerts') => void
}

const Navigation: React.FC<NavigationProps> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'results', label: 'Results', icon: Microscope },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: '3d-viz', label: '3D View', icon: Box },
    { id: 'maintenance', label: 'Maintenance', icon: Wrench },
    { id: 'classification', label: 'AI Classification', icon: Brain },
    { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
  ] as const

  return (
    <nav className="max-w-7xl mx-auto px-6 py-4">
      <div className="glass rounded-2xl p-2">
        <div className="flex flex-wrap gap-2">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={`tab-button px-6 py-3 rounded-xl transition-all font-medium flex items-center space-x-2 ${
                activeTab === id ? 'tab-active' : 'tab-inactive'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default Navigation
