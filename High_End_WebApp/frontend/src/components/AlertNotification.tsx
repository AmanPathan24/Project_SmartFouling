import React, { useState, useEffect } from 'react';
import { AlertTriangle, X, CheckCircle, Clock, ExternalLink } from 'lucide-react';
import { useApi } from '../contexts/ApiContext';

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

interface AlertNotificationProps {
  onDismiss?: (alertId: string) => void;
  autoHide?: boolean;
  autoHideDelay?: number;
}

const AlertNotification: React.FC<AlertNotificationProps> = ({
  onDismiss,
  autoHide = true,
  autoHideDelay = 10000
}) => {
  const [alerts, setAlerts] = useState<InvasiveAlert[]>([]);
  const [isVisible, setIsVisible] = useState(false);
  const [currentAlertIndex, setCurrentAlertIndex] = useState(0);
  const { api } = useApi();

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'bg-red-600 border-red-500 text-white';
      case 'high':
        return 'bg-orange-600 border-orange-500 text-white';
      case 'medium':
        return 'bg-yellow-600 border-yellow-500 text-white';
      case 'low':
        return 'bg-blue-600 border-blue-500 text-white';
      default:
        return 'bg-gray-600 border-gray-500 text-white';
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

  const fetchAlerts = async () => {
    try {
      const response = await api.get('/alerts/invasive-species');
      const activeAlerts = response.data.alerts.filter((alert: InvasiveAlert) => 
        alert.status === 'active' && alert.requires_immediate_action
      );
      
      if (activeAlerts.length > 0) {
        setAlerts(activeAlerts);
        setIsVisible(true);
      }
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await api.post(`/alerts/${alertId}/acknowledge`);
      setAlerts(prev => prev.filter(alert => alert.id !== alertId));
      
      if (alerts.length <= 1) {
        setIsVisible(false);
        setCurrentAlertIndex(0);
      } else {
        setCurrentAlertIndex(prev => (prev + 1) % alerts.length);
      }
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const dismissAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
    onDismiss?.(alertId);
    
    if (alerts.length <= 1) {
      setIsVisible(false);
      setCurrentAlertIndex(0);
    } else {
      setCurrentAlertIndex(prev => (prev + 1) % alerts.length);
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  useEffect(() => {
    // Fetch alerts every 30 seconds
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (alerts.length > 0 && autoHide) {
      const timer = setTimeout(() => {
        setCurrentAlertIndex(prev => (prev + 1) % alerts.length);
      }, autoHideDelay);

      return () => clearTimeout(timer);
    }
  }, [alerts.length, autoHide, autoHideDelay, currentAlertIndex]);

  if (!isVisible || alerts.length === 0) {
    return null;
  }

  const currentAlert = alerts[currentAlertIndex];

  return (
    <div className="fixed top-4 right-4 z-50 max-w-md">
      <div className={`rounded-lg shadow-2xl border-2 ${getRiskLevelColor(currentAlert.risk_level)} animate-pulse`}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/20">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">{getRiskLevelIcon(currentAlert.risk_level)}</span>
            <div>
              <h3 className="font-bold text-lg">Invasive Species Alert</h3>
              <p className="text-sm opacity-90">
                {currentAlert.risk_level.toUpperCase()} RISK
              </p>
            </div>
          </div>
          <button
            onClick={() => dismissAlert(currentAlert.id)}
            className="text-white/80 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Alert Content */}
        <div className="p-4">
          <div className="mb-3">
            <h4 className="font-semibold text-lg">{currentAlert.species_name}</h4>
            <p className="text-sm opacity-90">{currentAlert.common_name}</p>
            <p className="text-xs opacity-75 mt-1">
              Confidence: {(currentAlert.confidence * 100).toFixed(1)}%
            </p>
          </div>

          <div className="mb-3">
            <p className="text-sm">{currentAlert.impact}</p>
          </div>

          {currentAlert.recommended_actions.length > 0 && (
            <div className="mb-3">
              <p className="text-xs font-semibold mb-1">Immediate Actions:</p>
              <ul className="text-xs space-y-1">
                {currentAlert.recommended_actions.slice(0, 2).map((action, index) => (
                  <li key={index} className="flex items-start space-x-1">
                    <span className="mt-0.5">•</span>
                    <span>{action}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex items-center justify-between text-xs opacity-75">
            <div className="flex items-center space-x-1">
              <Clock className="w-3 h-3" />
              <span>{formatTime(currentAlert.detection_time)}</span>
            </div>
            {currentAlert.session_name && (
              <div className="flex items-center space-x-1">
                <ExternalLink className="w-3 h-3" />
                <span>{currentAlert.session_name}</span>
              </div>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex border-t border-white/20">
          <button
            onClick={() => acknowledgeAlert(currentAlert.id)}
            className="flex-1 flex items-center justify-center space-x-2 py-3 hover:bg-white/10 transition-colors"
          >
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm font-medium">Acknowledge</span>
          </button>
          {alerts.length > 1 && (
            <button
              onClick={() => setCurrentAlertIndex(prev => (prev + 1) % alerts.length)}
              className="flex-1 py-3 hover:bg-white/10 transition-colors text-sm"
            >
              Next ({alerts.length})
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default AlertNotification;
