import React, { useState, useEffect } from 'react';
import { useApi } from '../contexts/ApiContext';
import { AlertTriangle, Shield, Clock, CheckCircle, Eye, ExternalLink } from 'lucide-react';

interface InvasiveAlert {
  id: string;
  type: string;
  risk_level: 'critical' | 'high' | 'medium' | 'low';
  species_name: string;
  common_name: string;
  confidence: number;
  impact: string;
  recommended_actions: string[];
  detection_time: string;
  session_id?: string;
  session_name?: string;
  requires_immediate_action: boolean;
  status: string;
}

interface AlertSummary {
  total_alerts: number;
  critical_alerts: number;
  high_risk_alerts: number;
  unique_species: number;
  immediate_action_required: boolean;
  risk_distribution: Record<string, number>;
  top_species: Array<[string, number]>;
}

const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<InvasiveAlert[]>([]);
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>('all');
  const [selectedAlert, setSelectedAlert] = useState<InvasiveAlert | null>(null);
  const { api } = useApi();

  const fetchAlerts = async () => {
    try {
      setIsLoading(true);
      const response = await api.get('/alerts/invasive-species');
      setAlerts(response.data.alerts || []);
      setSummary(response.data.summary || null);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await api.post(`/alerts/${alertId}/acknowledge`);
      setAlerts(prev => prev.filter(alert => alert.id !== alertId));
      fetchAlerts(); // Refresh data
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'bg-red-100 border-red-500 text-red-800';
      case 'high':
        return 'bg-orange-100 border-orange-500 text-orange-800';
      case 'medium':
        return 'bg-yellow-100 border-yellow-500 text-yellow-800';
      case 'low':
        return 'bg-blue-100 border-blue-500 text-blue-800';
      default:
        return 'bg-gray-100 border-gray-500 text-gray-800';
    }
  };

  const getRiskLevelIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return '🚨';
      case 'high':
        return '⚠️';
      case 'medium':
        return '⚡';
      case 'low':
        return 'ℹ️';
      default:
        return '📋';
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const filteredAlerts = alerts.filter(alert => 
    selectedRiskLevel === 'all' || alert.risk_level === selectedRiskLevel
  );

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading invasive species alerts...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Invasive Species Alerts</h2>
        <button
          onClick={fetchAlerts}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg shadow border-l-4 border-red-500">
            <div className="flex items-center">
              <AlertTriangle className="w-8 h-8 text-red-500 mr-3" />
              <div>
                <p className="text-sm text-gray-600">Critical Alerts</p>
                <p className="text-2xl font-bold text-red-600">{summary.critical_alerts}</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow border-l-4 border-orange-500">
            <div className="flex items-center">
              <Shield className="w-8 h-8 text-orange-500 mr-3" />
              <div>
                <p className="text-sm text-gray-600">High Risk</p>
                <p className="text-2xl font-bold text-orange-600">{summary.high_risk_alerts}</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow border-l-4 border-blue-500">
            <div className="flex items-center">
              <Eye className="w-8 h-8 text-blue-500 mr-3" />
              <div>
                <p className="text-sm text-gray-600">Total Alerts</p>
                <p className="text-2xl font-bold text-blue-600">{summary.total_alerts}</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow border-l-4 border-green-500">
            <div className="flex items-center">
              <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
              <div>
                <p className="text-sm text-gray-600">Unique Species</p>
                <p className="text-2xl font-bold text-green-600">{summary.unique_species}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filter Controls */}
      <div className="flex items-center space-x-4">
        <label className="text-sm font-medium text-gray-700">Filter by Risk Level:</label>
        <select
          value={selectedRiskLevel}
          onChange={(e) => setSelectedRiskLevel(e.target.value)}
          className="px-3 py-1 border border-gray-300 rounded-md text-sm"
        >
          <option value="all">All Alerts</option>
          <option value="critical">Critical</option>
          <option value="high">High Risk</option>
          <option value="medium">Medium Risk</option>
          <option value="low">Low Risk</option>
        </select>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <Shield className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Alerts Found</h3>
            <p className="text-gray-600">
              {selectedRiskLevel === 'all' 
                ? 'No invasive species alerts detected.' 
                : `No ${selectedRiskLevel} risk alerts found.`}
            </p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`bg-white rounded-lg shadow border-l-4 ${getRiskLevelColor(alert.risk_level).split(' ')[1]}`}
            >
              <div className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <span className="text-2xl">{getRiskLevelIcon(alert.risk_level)}</span>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="font-semibold text-lg">{alert.species_name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskLevelColor(alert.risk_level)}`}>
                          {alert.risk_level.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-gray-600 mb-2">{alert.common_name}</p>
                      <p className="text-sm text-gray-700 mb-3">{alert.impact}</p>
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <div className="flex items-center space-x-1">
                          <Clock className="w-4 h-4" />
                          <span>{formatTime(alert.detection_time)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <span>Confidence: {(alert.confidence * 100).toFixed(1)}%</span>
                        </div>
                        {alert.session_name && (
                          <div className="flex items-center space-x-1">
                            <ExternalLink className="w-4 h-4" />
                            <span>{alert.session_name}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setSelectedAlert(alert)}
                      className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                    >
                      View Details
                    </button>
                    <button
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
                    >
                      Acknowledge
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Alert Details Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">Alert Details</h3>
                <button
                  onClick={() => setSelectedAlert(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-lg">{selectedAlert.species_name}</h4>
                  <p className="text-gray-600">{selectedAlert.common_name}</p>
                </div>
                
                <div>
                  <h5 className="font-medium mb-2">Impact</h5>
                  <p className="text-gray-700">{selectedAlert.impact}</p>
                </div>
                
                <div>
                  <h5 className="font-medium mb-2">Recommended Actions</h5>
                  <ul className="space-y-1">
                    {selectedAlert.recommended_actions.map((action, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <span className="text-blue-500 mt-1">•</span>
                        <span className="text-gray-700">{action}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="flex justify-end space-x-3 pt-4 border-t">
                  <button
                    onClick={() => setSelectedAlert(null)}
                    className="px-4 py-2 text-gray-600 border border-gray-300 rounded hover:bg-gray-50"
                  >
                    Close
                  </button>
                  <button
                    onClick={() => {
                      acknowledgeAlert(selectedAlert.id);
                      setSelectedAlert(null);
                    }}
                    className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Acknowledge Alert
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertsPanel;
