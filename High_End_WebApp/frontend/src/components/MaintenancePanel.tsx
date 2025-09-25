import React, { useState } from 'react'
import { Wrench, Clock, DollarSign, AlertTriangle, CheckCircle, Plus, Calendar, Filter, Search } from 'lucide-react'
import { useAnalytics, MaintenanceTask } from '../contexts/AnalyticsContext'
import { useSession } from '../contexts/SessionContext'
import toast from 'react-hot-toast'

const MaintenancePanel: React.FC = () => {
  const { maintenanceTasks, isLoading, addMaintenanceTask, updateMaintenanceTask } = useAnalytics()
  const { currentSession } = useSession()
  const [showAddForm, setShowAddForm] = useState(false)
  const [filterPriority, setFilterPriority] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'text-red-400 bg-red-500/20 border-red-500/30'
      case 'high': return 'text-orange-400 bg-orange-500/20 border-orange-500/30'
      case 'medium': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      case 'low': return 'text-green-400 bg-green-500/20 border-green-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'critical': return <AlertTriangle className="w-4 h-4" />
      case 'high': return <AlertTriangle className="w-4 h-4" />
      case 'medium': return <AlertTriangle className="w-4 h-4" />
      case 'low': return <CheckCircle className="w-4 h-4" />
      default: return <CheckCircle className="w-4 h-4" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 bg-green-500/20 border-green-500/30'
      case 'in_progress': return 'text-blue-400 bg-blue-500/20 border-blue-500/30'
      case 'pending': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  const filteredTasks = maintenanceTasks.filter(task => {
    const matchesPriority = filterPriority === 'all' || task.priority === filterPriority
    const matchesSearch = task.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         task.description.toLowerCase().includes(searchTerm.toLowerCase())
    return matchesPriority && matchesSearch
  })

  const handleStatusUpdate = async (taskId: number, newStatus: string) => {
    try {
      await updateMaintenanceTask(taskId, { status: newStatus })
      toast.success('Task status updated successfully!')
    } catch (error) {
      toast.error('Failed to update task status')
    }
  }

  const handleAddTask = async (taskData: Omit<MaintenanceTask, 'id' | 'created_at'>) => {
    try {
      await addMaintenanceTask(taskData)
      setShowAddForm(false)
      toast.success('Maintenance task added successfully!')
    } catch (error) {
      toast.error('Failed to add maintenance task')
    }
  }

  const calculateTotalCost = () => {
    return maintenanceTasks
      .filter(task => task.status !== 'completed')
      .reduce((sum, task) => sum + task.estimated_cost, 0)
  }

  const getUpcomingTasks = () => {
    const now = new Date()
    const nextWeek = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000)
    
    return maintenanceTasks.filter(task => {
      const taskDate = new Date(task.recommended_date)
      return taskDate <= nextWeek && task.status === 'pending'
    })
  }

  if (isLoading) {
    return (
      <div className="glass-card p-12 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
        <p className="text-gray-400">Loading maintenance tasks...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold gradient-text flex items-center">
              <Wrench className="w-6 h-6 mr-3" />
              Maintenance Management
            </h2>
            <p className="text-gray-400 mt-1">
              Schedule and track maintenance tasks based on fouling analysis
            </p>
          </div>
          <button
            onClick={() => setShowAddForm(true)}
            className="glass-button px-6 py-3 font-medium flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Add Task</span>
          </button>
        </div>

        {/* Summary Cards */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="glass p-4 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Total Tasks</span>
              <Wrench className="w-4 h-4 text-blue-400" />
            </div>
            <span className="text-2xl font-bold text-blue-400">
              {maintenanceTasks.length}
            </span>
          </div>
          
          <div className="glass p-4 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Pending</span>
              <Clock className="w-4 h-4 text-yellow-400" />
            </div>
            <span className="text-2xl font-bold text-yellow-400">
              {maintenanceTasks.filter(t => t.status === 'pending').length}
            </span>
          </div>
          
          <div className="glass p-4 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">In Progress</span>
              <AlertTriangle className="w-4 h-4 text-orange-400" />
            </div>
            <span className="text-2xl font-bold text-orange-400">
              {maintenanceTasks.filter(t => t.status === 'in_progress').length}
            </span>
          </div>
          
          <div className="glass p-4 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Total Cost</span>
              <DollarSign className="w-4 h-4 text-green-400" />
            </div>
            <span className="text-2xl font-bold text-green-400">
              ${calculateTotalCost().toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="glass-card p-6">
        <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Filter className="w-4 h-4 text-gray-400" />
              <select 
                value={filterPriority}
                onChange={(e) => setFilterPriority(e.target.value)}
                className="glass px-3 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-400/50"
              >
                <option value="all">All Priorities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Search className="w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search tasks..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="glass px-3 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-400/50"
            />
          </div>
        </div>
      </div>

      {/* Upcoming Tasks Alert */}
      {getUpcomingTasks().length > 0 && (
        <div className="glass-card p-6 border border-yellow-500/30">
          <div className="flex items-center space-x-3 mb-4">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            <h3 className="text-lg font-semibold text-yellow-400">Upcoming Tasks (Next 7 Days)</h3>
          </div>
          <div className="space-y-2">
            {getUpcomingTasks().map(task => (
              <div key={task.id} className="flex items-center justify-between glass p-3 rounded-lg">
                <div>
                  <h4 className="font-medium text-white">{task.title}</h4>
                  <p className="text-sm text-gray-400">
                    Due: {new Date(task.recommended_date).toLocaleDateString()}
                  </p>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full border flex items-center space-x-1 ${getPriorityColor(task.priority)}`}>
                  {getPriorityIcon(task.priority)}
                  <span>{task.priority}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tasks List */}
      <div className="space-y-4">
        {filteredTasks.map(task => (
          <div key={task.id} className="glass-card p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="text-lg font-semibold text-white">{task.title}</h3>
                  <span className={`px-2 py-1 text-xs rounded-full border flex items-center space-x-1 ${getPriorityColor(task.priority)}`}>
                    {getPriorityIcon(task.priority)}
                    <span>{task.priority}</span>
                  </span>
                  <span className={`px-2 py-1 text-xs rounded-full border ${getStatusColor(task.status || 'pending')}`}>
                    {task.status || 'pending'}
                  </span>
                </div>
                <p className="text-gray-300 mb-3">{task.description}</p>
                
                <div className="grid md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Estimated Cost:</span>
                    <span className="ml-2 font-semibold text-green-400">
                      ${task.estimated_cost.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Duration:</span>
                    <span className="ml-2 font-semibold text-blue-400">
                      {task.estimated_duration}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Recommended Date:</span>
                    <span className="ml-2 font-semibold text-purple-400">
                      {new Date(task.recommended_date).toLocaleDateString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Coverage:</span>
                    <span className="ml-2 font-semibold text-yellow-400">
                      {task.coverage.toFixed(1)}%
                    </span>
                  </div>
                </div>
                
                <div className="mt-3">
                  <span className="text-gray-400 text-sm">Species: </span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {task.species.map((species, index) => (
                      <span key={index} className="px-2 py-1 text-xs bg-blue-500/20 text-blue-300 rounded-full">
                        {species}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col space-y-2 ml-4">
                <select
                  value={task.status || 'pending'}
                  onChange={(e) => handleStatusUpdate(task.id, e.target.value)}
                  className="glass px-3 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-400/50"
                >
                  <option value="pending">Pending</option>
                  <option value="in_progress">In Progress</option>
                  <option value="completed">Completed</option>
                </select>
                
                <button className="glass-button px-4 py-2 text-sm flex items-center space-x-2">
                  <Calendar className="w-3 h-3" />
                  <span>Schedule</span>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Add Task Form Modal */}
      {showAddForm && (
        <AddTaskForm
          onClose={() => setShowAddForm(false)}
          onSubmit={handleAddTask}
          currentSession={currentSession}
        />
      )}
    </div>
  )
}

// Add Task Form Component
const AddTaskForm: React.FC<{
  onClose: () => void
  onSubmit: (task: Omit<MaintenanceTask, 'id' | 'created_at'>) => void
  currentSession: any
}> = ({ onClose, onSubmit, currentSession }) => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    priority: 'medium' as const,
    estimated_cost: 0,
    estimated_duration: '',
    recommended_date: '',
    species: [] as string[],
    coverage: 0
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
  }

  const handleSpeciesChange = (species: string, checked: boolean) => {
    if (checked) {
      setFormData(prev => ({ ...prev, species: [...prev.species, species] }))
    } else {
      setFormData(prev => ({ ...prev, species: prev.species.filter(s => s !== species) }))
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-md flex items-center justify-center z-50">
      <div className="glass-card p-8 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <h3 className="text-xl font-bold gradient-text mb-6">Add Maintenance Task</h3>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">Task Title</label>
            <input
              type="text"
              value={formData.title}
              onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
              className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">Description</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50 h-24 resize-none"
              required
            />
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-300 mb-2">Priority</label>
              <select
                value={formData.priority}
                onChange={(e) => setFormData(prev => ({ ...prev, priority: e.target.value as any }))}
                className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-semibold text-gray-300 mb-2">Estimated Cost ($)</label>
              <input
                type="number"
                value={formData.estimated_cost}
                onChange={(e) => setFormData(prev => ({ ...prev, estimated_cost: parseInt(e.target.value) }))}
                className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50"
                required
              />
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-300 mb-2">Duration</label>
              <input
                type="text"
                value={formData.estimated_duration}
                onChange={(e) => setFormData(prev => ({ ...prev, estimated_duration: e.target.value }))}
                placeholder="e.g., 4 hours"
                className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-semibold text-gray-300 mb-2">Recommended Date</label>
              <input
                type="date"
                value={formData.recommended_date}
                onChange={(e) => setFormData(prev => ({ ...prev, recommended_date: e.target.value }))}
                className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50"
                required
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">Fouling Species</label>
            <div className="grid grid-cols-2 gap-2">
              {['Barnacles', 'Mussels', 'Seaweed', 'Sponges', 'Anemones', 'Tunicates'].map(species => (
                <label key={species} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={formData.species.includes(species)}
                    onChange={(e) => handleSpeciesChange(species, e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm text-gray-300">{species}</span>
                </label>
              ))}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">Coverage Percentage</label>
            <input
              type="number"
              min="0"
              max="100"
              step="0.1"
              value={formData.coverage}
              onChange={(e) => setFormData(prev => ({ ...prev, coverage: parseFloat(e.target.value) }))}
              className="w-full glass px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400/50"
              required
            />
          </div>
          
          <div className="flex justify-end space-x-4">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-3 text-gray-300 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="glass-button px-6 py-3 font-medium"
            >
              Add Task
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default MaintenancePanel
