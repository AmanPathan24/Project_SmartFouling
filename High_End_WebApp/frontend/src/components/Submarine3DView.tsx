import React, { useEffect, useRef, useState } from 'react';
import { useApi } from '../contexts/ApiContext';
import { useSession } from '../contexts/SessionContext';
import { Loader2, RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface AreaRatioData {
  [zoneName: string]: number;
}

interface Submarine3DViewProps {
  className?: string;
}

const Submarine3DView: React.FC<Submarine3DViewProps> = ({ className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const submarineRef = useRef<THREE.Group | null>(null);
  const animationIdRef = useRef<number | null>(null);
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [areaRatios, setAreaRatios] = useState<AreaRatioData>({});
  const [isControlsEnabled, setIsControlsEnabled] = useState(true);
  
  const { api } = useApi();
  const { currentSession } = useSession();

  // Heat color function: 0..1 -> green->yellow->red
  const getHeatColor = (value: number): THREE.Color => {
    const t = Math.max(0, Math.min(1, value));
    if (t <= 0.5) {
      const k = t / 0.5; // 0..1
      return new THREE.Color(`rgb(${Math.round(255 * k)},255,0)`);
    } else {
      const k = (t - 0.5) / 0.5;
      return new THREE.Color(`rgb(255,${Math.round(255 * (1 - k))},0)`);
    }
  };

  // Initialize Three.js scene
  const initializeScene = () => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x02040a);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      50,
      container.clientWidth / container.clientHeight,
      0.1,
      200
    );
    camera.position.set(0, 2.2, 6);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    container.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9);
    directionalLight.position.set(5, 10, 7);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    
    scene.add(ambientLight, directionalLight);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.enableRotate = true;
    controls.enablePan = true;
    controlsRef.current = controls;
  };

  // Load submarine model
  const loadSubmarineModel = async () => {
    if (!sceneRef.current) return;

    try {
      const loader = new GLTFLoader();

      loader.load(
        '/api/models/submarine.glb',
        (gltf) => {
          const submarine = gltf.scene;
          submarineRef.current = submarine;

          // Apply materials to each mesh
          submarine.traverse((obj) => {
            if (!obj.isMesh) return;

            const zoneName = obj.name;
            const hasValue = Object.prototype.hasOwnProperty.call(areaRatios, zoneName);
            const value = hasValue ? areaRatios[zoneName] : -1;

            // Base color: green if missing, heat color if present
            const baseColor = (value < 0) ? new THREE.Color('#2ecc71') : getHeatColor(value);

            // Main material with transparency
            const mainMaterial = new THREE.MeshPhysicalMaterial({
              color: baseColor,
              roughness: 0.8,
              metalness: 0.0,
              transparent: true,
              opacity: 0.65,
              transmission: 0.0,
              depthWrite: true,
            });

            // Wireframe overlay
            const wireframeMaterial = new THREE.MeshStandardMaterial({
              color: new THREE.Color(0x11c5d9),
              wireframe: true,
              transparent: true,
              opacity: 0.28,
            });

            // Create main mesh
            const mainMesh = new THREE.Mesh(obj.geometry, mainMaterial);
            mainMesh.position.copy(obj.position);
            mainMesh.quaternion.copy(obj.quaternion);
            mainMesh.scale.copy(obj.scale);
            mainMesh.renderOrder = 0;
            mainMesh.castShadow = true;
            mainMesh.receiveShadow = true;

            // Create wireframe overlay
            const wireframeMesh = new THREE.Mesh(obj.geometry, wireframeMaterial);
            wireframeMesh.position.copy(obj.position);
            wireframeMesh.quaternion.copy(obj.quaternion);
            wireframeMesh.scale.copy(obj.scale);
            wireframeMesh.renderOrder = 1;

            // Hide original and add new meshes
            obj.visible = false;
            if (obj.parent) {
              obj.parent.add(mainMesh);
              obj.parent.add(wireframeMesh);
            }
          });

          sceneRef.current.add(submarine);
          fitView(submarine);
          setIsLoading(false);
        },
        undefined,
        (error) => {
          console.error('Failed to load submarine model:', error);
          setError('Failed to load submarine model');
          setIsLoading(false);
        }
      );
    } catch (error) {
      console.error('Error loading submarine model:', error);
      setError('Error loading 3D model');
      setIsLoading(false);
    }
  };

  // Fit view to model
  const fitView = (object: THREE.Object3D) => {
    if (!cameraRef.current || !controlsRef.current) return;

    const box = new THREE.Box3().setFromObject(object);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = cameraRef.current.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / (2 * Math.tan(fov / 2)));
    cameraZ *= 1.4;
    
    cameraRef.current.position.set(center.x, center.y + maxDim * 0.15, center.z + cameraZ);
    cameraRef.current.lookAt(center);
    controlsRef.current.target.copy(center);
    controlsRef.current.update();
  };

  // Animation loop
  const animate = () => {
    if (controlsRef.current) {
      controlsRef.current.update();
    }
    
    if (rendererRef.current && sceneRef.current && cameraRef.current) {
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    }
    
    animationIdRef.current = requestAnimationFrame(animate);
  };

  // Load area ratio data
  const loadAreaRatios = async () => {
    if (!currentSession) {
      // Use mock data for demonstration
      setAreaRatios({
        "middle_left_bottom": 0.9,
        "front_right_top": 0.25,
        "rear_center": 0.6,
        "top_surface": 0.15,
        "bottom_hull": 0.8,
      });
      return;
    }

    try {
      const response = await api.get(`/sessions/${currentSession.session_id}/area-ratios`);
      setAreaRatios(response.data);
    } catch (error) {
      console.error('Failed to load area ratios:', error);
      // Use mock data as fallback
      setAreaRatios({
        "middle_left_bottom": 0.9,
        "front_right_top": 0.25,
        "rear_center": 0.6,
        "top_surface": 0.15,
        "bottom_hull": 0.8,
      });
    }
  };

  // Control functions
  const resetView = () => {
    if (submarineRef.current) {
      fitView(submarineRef.current);
    }
  };

  const toggleControls = () => {
    if (controlsRef.current) {
      const enabled = !isControlsEnabled;
      controlsRef.current.enabled = enabled;
      setIsControlsEnabled(enabled);
    }
  };

  const zoomIn = () => {
    if (cameraRef.current && controlsRef.current) {
      const direction = new THREE.Vector3();
      cameraRef.current.getWorldDirection(direction);
      cameraRef.current.position.addScaledVector(direction, -1);
      controlsRef.current.update();
    }
  };

  const zoomOut = () => {
    if (cameraRef.current && controlsRef.current) {
      const direction = new THREE.Vector3();
      cameraRef.current.getWorldDirection(direction);
      cameraRef.current.position.addScaledVector(direction, 1);
      controlsRef.current.update();
    }
  };

  // Handle resize
  const handleResize = () => {
    if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(width, height);
  };

  // Initialize and cleanup
  useEffect(() => {
    try {
      initializeScene();
      loadAreaRatios();
    } catch (error) {
      console.error('Error initializing 3D scene:', error);
      setError('Failed to initialize 3D scene');
      setIsLoading(false);
    }

    const handleResizeDebounced = () => {
      setTimeout(handleResize, 100);
    };

    window.addEventListener('resize', handleResizeDebounced);

    return () => {
      window.removeEventListener('resize', handleResizeDebounced);
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current && containerRef.current && containerRef.current.contains(rendererRef.current.domElement)) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
    };
  }, []);

  // Load model when area ratios are available
  useEffect(() => {
    if (Object.keys(areaRatios).length > 0) {
      loadSubmarineModel();
    }
  }, [areaRatios]);

  // Start animation loop
  useEffect(() => {
    animate();
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    };
  }, []);

  return (
    <div className={`relative ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-800">3D Submarine Hull Analysis</h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={resetView}
            className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
          <button
            onClick={zoomIn}
            className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
          <button
            onClick={zoomOut}
            className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-5 h-5" />
          </button>
          <button
            onClick={toggleControls}
            className={`p-2 rounded-lg transition-colors ${
              isControlsEnabled 
                ? 'text-blue-600 bg-blue-50' 
                : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
            }`}
            title="Toggle Controls"
          >
            <Maximize2 className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* 3D Viewer */}
      <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl overflow-hidden shadow-2xl">
        <div 
          ref={containerRef} 
          className="w-full h-96 md:h-[500px] relative"
          style={{ minHeight: '400px' }}
        />
        
        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-slate-900 bg-opacity-75 flex items-center justify-center">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-2" />
              <p className="text-white text-sm">Loading 3D Model...</p>
            </div>
          </div>
        )}

        {/* Error Overlay */}
        {error && (
          <div className="absolute inset-0 bg-slate-900 bg-opacity-75 flex items-center justify-center">
            <div className="text-center">
              <p className="text-red-400 text-sm mb-2">Error Loading Model</p>
              <p className="text-gray-300 text-xs">{error}</p>
              <button 
                onClick={() => {
                  setError(null);
                  setIsLoading(true);
                  loadAreaRatios();
                }}
                className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Retry
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 p-4 bg-white rounded-lg shadow-sm">
        <h3 className="font-semibold text-gray-800 mb-3">Fouling Intensity Legend</h3>
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span className="text-sm text-gray-600">Clean (0-0.3)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-yellow-500 rounded"></div>
            <span className="text-sm text-gray-600">Moderate (0.3-0.7)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span className="text-sm text-gray-600">High (0.7-1.0)</span>
          </div>
        </div>
        
        {/* Area Data Table */}
        {Object.keys(areaRatios).length > 0 && (
          <div className="mt-4">
            <h4 className="font-medium text-gray-800 mb-2">Area Analysis</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
              {Object.entries(areaRatios).map(([zone, ratio]) => (
                <div key={zone} className="flex justify-between items-center">
                  <span className="text-gray-600 capitalize">{zone.replace(/_/g, ' ')}</span>
                  <span className={`font-medium ${
                    ratio > 0.7 ? 'text-red-600' : 
                    ratio > 0.3 ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                    {(ratio * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-800">
          <strong>Controls:</strong> Click and drag to rotate • Scroll to zoom • Right-click and drag to pan
        </p>
      </div>
    </div>
  );
};

export default Submarine3DView;