import React, { useEffect, useRef, useState } from 'react';
import { useApi } from '../contexts/ApiContext';
import { useSession } from '../contexts/SessionContext';

const Debug3DView: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [debugInfo, setDebugInfo] = useState<string[]>([]);
  const [threeStatus, setThreeStatus] = useState<string>('Not loaded');
  const [modelStatus, setModelStatus] = useState<string>('Not tested');
  const [apiStatus, setApiStatus] = useState<string>('Not tested');
  
  const { api } = useApi();
  const { currentSession } = useSession();

  const addDebugInfo = (info: string) => {
    setDebugInfo(prev => [...prev, `${new Date().toLocaleTimeString()}: ${info}`]);
  };

  // Test Three.js loading
  const testThreeJS = async () => {
    try {
      addDebugInfo('Testing Three.js import...');
      const THREE = await import('three');
      addDebugInfo(`Three.js loaded successfully. Version: ${THREE.REVISION}`);
      setThreeStatus('Loaded successfully');
      
      // Test basic scene creation
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer();
      addDebugInfo('Basic Three.js scene created successfully');
      
    } catch (error) {
      addDebugInfo(`Three.js import failed: ${error}`);
      setThreeStatus('Failed to load');
    }
  };

  // Test model endpoint
  const testModelEndpoint = async () => {
    try {
      addDebugInfo('Testing submarine model endpoint...');
      const response = await fetch('/api/models/submarine.glb', { method: 'HEAD' });
      if (response.ok) {
        addDebugInfo(`Model endpoint accessible. Status: ${response.status}`);
        setModelStatus('Accessible');
      } else {
        addDebugInfo(`Model endpoint failed. Status: ${response.status}`);
        setModelStatus('Failed');
      }
    } catch (error) {
      addDebugInfo(`Model endpoint error: ${error}`);
      setModelStatus('Error');
    }
  };

  // Test API
  const testAPI = async () => {
    try {
      addDebugInfo('Testing area ratios API...');
      const response = await api.get('/sessions/test_session/area-ratios');
      addDebugInfo(`Area ratios API successful. Data: ${JSON.stringify(response.data)}`);
      setApiStatus('Working');
    } catch (error) {
      addDebugInfo(`Area ratios API failed: ${error}`);
      setApiStatus('Failed');
    }
  };

  // Test session context
  const testSessionContext = () => {
    addDebugInfo(`Current session: ${currentSession ? currentSession.session_id : 'None'}`);
  };

  // Run all tests
  const runAllTests = async () => {
    setDebugInfo([]);
    addDebugInfo('Starting debug tests...');
    
    testSessionContext();
    await testThreeJS();
    await testModelEndpoint();
    await testAPI();
    
    addDebugInfo('All tests completed');
  };

  useEffect(() => {
    runAllTests();
  }, []);

  return (
    <div className="relative p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">3D Debug Information</h2>
      
      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-semibold text-gray-800 mb-2">Three.js Status</h3>
          <p className={`text-sm ${threeStatus.includes('successfully') ? 'text-green-600' : 'text-red-600'}`}>
            {threeStatus}
          </p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-semibold text-gray-800 mb-2">Model Status</h3>
          <p className={`text-sm ${modelStatus === 'Accessible' ? 'text-green-600' : 'text-red-600'}`}>
            {modelStatus}
          </p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-semibold text-gray-800 mb-2">API Status</h3>
          <p className={`text-sm ${apiStatus === 'Working' ? 'text-green-600' : 'text-red-600'}`}>
            {apiStatus}
          </p>
        </div>
      </div>

      {/* Debug Log */}
      <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
        <h3 className="text-white mb-2">Debug Log:</h3>
        <div className="max-h-96 overflow-y-auto">
          {debugInfo.map((info, index) => (
            <div key={index} className="mb-1">{info}</div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-6 flex space-x-4">
        <button
          onClick={runAllTests}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Run All Tests
        </button>
        <button
          onClick={testThreeJS}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
        >
          Test Three.js
        </button>
        <button
          onClick={testModelEndpoint}
          className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
        >
          Test Model
        </button>
        <button
          onClick={testAPI}
          className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700"
        >
          Test API
        </button>
      </div>

      {/* Session Info */}
      <div className="mt-6 bg-white p-4 rounded-lg shadow">
        <h3 className="font-semibold text-gray-800 mb-2">Session Context</h3>
        <pre className="text-sm text-gray-600 bg-gray-100 p-2 rounded">
          {JSON.stringify(currentSession, null, 2)}
        </pre>
      </div>
    </div>
  );
};

export default Debug3DView;
