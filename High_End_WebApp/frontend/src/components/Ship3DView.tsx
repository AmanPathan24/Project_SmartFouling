import React, { useRef, useState, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Box, Sphere, Cylinder } from '@react-three/drei'
import { Box3, Vector3 } from 'three'
import { RotateCcw, ZoomIn, ZoomOut, Home, Play, Pause } from 'lucide-react'
import { useSession } from '../contexts/SessionContext'

// Ship Hull Component
const ShipHull: React.FC<{ foulingData?: any }> = ({ foulingData }) => {
  const meshRef = useRef<any>()
  const [hovered, setHovered] = useState(false)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.1
    }
  })

  // Generate fouling regions based on data
  const generateFoulingRegions = () => {
    if (!foulingData) return []
    
    const regions = []
    foulingData.forEach((detection: any, index: number) => {
      regions.push(
        <Sphere
          key={index}
          position={[
            (Math.random() - 0.5) * 8,
            (Math.random() - 0.5) * 3,
            (Math.random() - 0.5) * 15
          ]}
          args={[0.3 + detection.coverage_percentage * 0.02]}
        >
          <meshStandardMaterial 
            color={getFoulingColor(detection.species)} 
            emissive={getFoulingColor(detection.species)}
            emissiveIntensity={0.2}
            transparent
            opacity={0.8}
          />
        </Sphere>
      )
    })
    return regions
  }

  const getFoulingColor = (species: string) => {
    const colors: { [key: string]: string } = {
      'Barnacles': '#ff4444',
      'Mussels': '#44ff44',
      'Seaweed': '#4444ff',
      'Sponges': '#ffff44',
      'Anemones': '#ff44ff',
      'Tunicates': '#44ffff',
      'Other_Fouling': '#888888'
    }
    return colors[species] || '#888888'
  }

  return (
    <group ref={meshRef}>
      {/* Main Hull */}
      <Box
        args={[12, 4, 20]}
        position={[0, 0, 0]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial 
          color={hovered ? '#4a5568' : '#2d3748'} 
          metalness={0.8}
          roughness={0.2}
        />
      </Box>

      {/* Bow */}
      <Box
        args={[8, 3, 6]}
        position={[0, 0, 13]}
        rotation={[0, 0, 0]}
      >
        <meshStandardMaterial 
          color="#2d3748" 
          metalness={0.8}
          roughness={0.2}
        />
      </Box>

      {/* Stern */}
      <Box
        args={[10, 3.5, 4]}
        position={[0, 0, -13]}
      >
        <meshStandardMaterial 
          color="#2d3748" 
          metalness={0.8}
          roughness={0.2}
        />
      </Box>

      {/* Propeller */}
      <Cylinder
        args={[1, 1, 0.5]}
        position={[0, -1.5, -15]}
        rotation={[Math.PI / 2, 0, 0]}
      >
        <meshStandardMaterial 
          color="#1a202c" 
          metalness={0.9}
          roughness={0.1}
        />
      </Cylinder>

      {/* Rudder */}
      <Box
        args={[0.5, 3, 4]}
        position={[0, -1, -15]}
      >
        <meshStandardMaterial 
          color="#1a202c" 
          metalness={0.9}
          roughness={0.1}
        />
      </Box>

      {/* Fouling Regions */}
      {generateFoulingRegions()}

      {/* Region Labels */}
      <Text
        position={[6, 2, 0]}
        fontSize={0.5}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        Port Side
      </Text>
      <Text
        position={[-6, 2, 0]}
        fontSize={0.5}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        Starboard
      </Text>
      <Text
        position={[0, 2, 13]}
        fontSize={0.5}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        Bow
      </Text>
      <Text
        position={[0, 2, -13]}
        fontSize={0.5}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        Stern
      </Text>
    </group>
  )
}

// Lighting Setup
const Lighting: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <pointLight position={[-10, -10, -5]} intensity={0.5} color="#4a90e2" />
    </>
  )
}

// Controls Component
const ViewControls: React.FC<{
  onReset: () => void
  onZoomIn: () => void
  onZoomOut: () => void
  onHome: () => void
  autoRotate: boolean
  onToggleRotate: () => void
}> = ({ onReset, onZoomIn, onZoomOut, onHome, autoRotate, onToggleRotate }) => {
  return (
    <div className="flex justify-center space-x-4 mt-6">
      <button
        onClick={onToggleRotate}
        className="glass-button px-6 py-3 font-medium flex items-center space-x-2"
      >
        {autoRotate ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        <span>{autoRotate ? 'Pause' : 'Auto Rotate'}</span>
      </button>
      <button
        onClick={onZoomIn}
        className="glass-button px-6 py-3 font-medium flex items-center space-x-2"
      >
        <ZoomIn className="w-4 h-4" />
        <span>Zoom In</span>
      </button>
      <button
        onClick={onZoomOut}
        className="glass-button px-6 py-3 font-medium flex items-center space-x-2"
      >
        <ZoomOut className="w-4 h-4" />
        <span>Zoom Out</span>
      </button>
      <button
        onClick={onHome}
        className="glass-button px-6 py-3 font-medium flex items-center space-x-2"
      >
        <Home className="w-4 h-4" />
        <span>Reset View</span>
      </button>
    </div>
  )
}

// Legend Component
const FoulingLegend: React.FC = () => {
  const speciesColors = [
    { species: 'Barnacles', color: '#ff4444' },
    { species: 'Mussels', color: '#44ff44' },
    { species: 'Seaweed', color: '#4444ff' },
    { species: 'Sponges', color: '#ffff44' },
    { species: 'Anemones', color: '#ff44ff' },
    { species: 'Tunicates', color: '#44ffff' },
    { species: 'Other Fouling', color: '#888888' }
  ]

  return (
    <div className="absolute top-4 right-4 glass-card p-4 max-w-xs">
      <h4 className="font-semibold text-white mb-3">Fouling Species Legend</h4>
      <div className="space-y-2">
        {speciesColors.map((item, index) => (
          <div key={index} className="flex items-center space-x-3">
            <div 
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: item.color }}
            ></div>
            <span className="text-sm text-gray-300">{item.species}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// Main 3D View Component
const Ship3DView: React.FC = () => {
  const { currentSession } = useSession()
  const [autoRotate, setAutoRotate] = useState(true)
  const controlsRef = useRef<any>()
  const cameraRef = useRef<any>()

  // Extract fouling data from current session
  const foulingData = React.useMemo(() => {
    if (!currentSession?.analysis_results) return []
    
    return currentSession.analysis_results.flatMap(result => 
      result.detections.map(detection => ({
        species: detection.species,
        coverage_percentage: detection.coverage_percentage,
        confidence: detection.confidence,
        bbox: detection.bbox
      }))
    )
  }, [currentSession])

  const handleReset = () => {
    if (controlsRef.current) {
      controlsRef.current.reset()
    }
  }

  const handleZoomIn = () => {
    if (cameraRef.current) {
      cameraRef.current.zoom = Math.min(cameraRef.current.zoom * 1.2, 5)
      cameraRef.current.updateProjectionMatrix()
    }
  }

  const handleZoomOut = () => {
    if (cameraRef.current) {
      cameraRef.current.zoom = Math.max(cameraRef.current.zoom / 1.2, 0.5)
      cameraRef.current.updateProjectionMatrix()
    }
  }

  const handleHome = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(20, 10, 20)
      cameraRef.current.lookAt(0, 0, 0)
      controlsRef.current.update()
    }
  }

  const handleToggleRotate = () => {
    setAutoRotate(!autoRotate)
  }

  return (
    <div className="glass-card p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-2xl font-bold gradient-text flex items-center">
            <div className="w-6 h-6 mr-3 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-xs font-bold">3D</span>
            </div>
            3D Hull Visualization
          </h3>
          <p className="text-gray-400 mt-1">
            Interactive 3D model showing fouling distribution across ship hull
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 text-xs font-medium bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
            Interactive
          </span>
          <span className="px-3 py-1 text-xs font-medium bg-blue-500/20 text-blue-300 rounded-full border border-blue-500/30">
            Real-time
          </span>
        </div>
      </div>

      <div className="relative">
        {/* 3D Canvas */}
        <div className="three-container">
          <Canvas
            camera={{ position: [20, 10, 20], fov: 50 }}
            shadows
          >
            <Lighting />
            <ShipHull foulingData={foulingData} />
            <OrbitControls
              ref={controlsRef}
              autoRotate={autoRotate}
              autoRotateSpeed={0.5}
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              minDistance={10}
              maxDistance={50}
            />
          </Canvas>
        </div>

        {/* Legend */}
        <FoulingLegend />

        {/* Info Panel */}
        {foulingData.length > 0 && (
          <div className="absolute bottom-4 left-4 glass-card p-4 max-w-sm">
            <h4 className="font-semibold text-white mb-2">Detection Summary</h4>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-300">Total Detections:</span>
                <span className="text-white font-medium">{foulingData.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Species Found:</span>
                <span className="text-white font-medium">
                  {new Set(foulingData.map(d => d.species)).size}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Avg Coverage:</span>
                <span className="text-white font-medium">
                  {(foulingData.reduce((sum, d) => sum + d.coverage_percentage, 0) / foulingData.length).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <ViewControls
        onReset={handleReset}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onHome={handleHome}
        autoRotate={autoRotate}
        onToggleRotate={handleToggleRotate}
      />

      {/* Instructions */}
      <div className="mt-6 p-4 glass rounded-xl">
        <h4 className="font-semibold text-white mb-2">Controls:</h4>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <p><strong>Mouse:</strong> Left click + drag to rotate, right click + drag to pan, scroll to zoom</p>
            <p><strong>Touch:</strong> Single finger to rotate, two fingers to pan and zoom</p>
          </div>
          <div>
            <p><strong>Colors:</strong> Red regions indicate high fouling density</p>
            <p><strong>Size:</strong> Larger spheres represent higher coverage percentages</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Ship3DView
