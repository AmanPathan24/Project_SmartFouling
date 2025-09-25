"""
Database Service for Marine Biofouling Detection
Handles data persistence and retrieval
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Database:
    """Simple file-based database for storing session data and analytics"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.sessions_file = self.data_dir / "sessions.json"
        self.datasets_file = self.data_dir / "datasets.json"
        self.analytics_file = self.data_dir / "analytics.json"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize database files with empty structures"""
        if not self.sessions_file.exists():
            self._save_json(self.sessions_file, {})
        
        if not self.datasets_file.exists():
            self._initialize_datasets()
        
        if not self.analytics_file.exists():
            self._save_json(self.analytics_file, {})
    
    def _initialize_datasets(self):
        """Initialize with sample datasets"""
        datasets = {
            "hull_inspection_2024": {
                "name": "Hull Inspection 2024",
                "description": "Quarterly hull inspection images from 2024",
                "images": [
                    {"filename": "hull1_original.jpg", "path": "demo/hull1_original.jpg"},
                    {"filename": "hull2_original.jpg", "path": "demo/hull2_original.jpg"},
                    {"filename": "prop_original.jpg", "path": "demo/prop_original.jpg"},
                    {"filename": "starboard_original.jpg", "path": "demo/starboard_original.jpg"}
                ],
                "created_at": "2024-01-15T10:00:00Z"
            },
            "port_side_analysis": {
                "name": "Port Side Analysis",
                "description": "Detailed port side fouling analysis",
                "images": [
                    {"filename": "port_bow.jpg", "path": "demo/hull1_original.jpg"},
                    {"filename": "port_midship.jpg", "path": "demo/hull2_original.jpg"}
                ],
                "created_at": "2024-02-01T14:30:00Z"
            },
            "propeller_inspection": {
                "name": "Propeller Inspection",
                "description": "Propeller fouling assessment",
                "images": [
                    {"filename": "propeller_1.jpg", "path": "demo/prop_original.jpg"},
                    {"filename": "propeller_2.jpg", "path": "demo/prop_original.jpg"}
                ],
                "created_at": "2024-02-15T09:15:00Z"
            }
        }
        
        self._save_json(self.datasets_file, datasets)
    
    async def initialize(self):
        """Initialize database (async placeholder)"""
        logger.info("Database initialized")
    
    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON data from file"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict):
        """Save JSON data to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
    
    async def save_session(self, session_data: Dict[str, Any]):
        """Save session data"""
        sessions = self._load_json(self.sessions_file)
        session_id = session_data["session_id"]
        sessions[session_id] = session_data
        self._save_json(self.sessions_file, sessions)
        logger.info(f"Session {session_id} saved")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID"""
        sessions = self._load_json(self.sessions_file)
        return sessions.get(session_id)
    
    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions"""
        sessions = self._load_json(self.sessions_file)
        return list(sessions.values())
    
    async def add_images_to_session(self, session_id: str, images: List[Dict[str, Any]]):
        """Add images to a session"""
        sessions = self._load_json(self.sessions_file)
        if session_id in sessions:
            if "images" not in sessions[session_id]:
                sessions[session_id]["images"] = []
            sessions[session_id]["images"].extend(images)
            sessions[session_id]["updated_at"] = datetime.now().isoformat()
            self._save_json(self.sessions_file, sessions)
    
    async def update_session_images(self, session_id: str, images: List[Dict[str, Any]]):
        """Update session images"""
        sessions = self._load_json(self.sessions_file)
        if session_id in sessions:
            sessions[session_id]["images"] = images
            sessions[session_id]["updated_at"] = datetime.now().isoformat()
            self._save_json(self.sessions_file, sessions)
    
    async def update_session_results(self, session_id: str, results: Dict[str, Any]):
        """Update session with analysis results"""
        sessions = self._load_json(self.sessions_file)
        if session_id in sessions:
            sessions[session_id].update(results)
            sessions[session_id]["updated_at"] = datetime.now().isoformat()
            self._save_json(self.sessions_file, sessions)
    
    async def get_datasets(self) -> List[Dict[str, Any]]:
        """Get all available datasets"""
        datasets = self._load_json(self.datasets_file)
        return [
            {
                "name": name,
                "display_name": data["name"],
                "description": data["description"],
                "image_count": len(data["images"]),
                "created_at": data["created_at"]
            }
            for name, data in datasets.items()
        ]
    
    async def get_dataset_images(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get images from a specific dataset"""
        datasets = self._load_json(self.datasets_file)
        if dataset_name in datasets:
            return datasets[dataset_name]["images"]
        return []
    
    async def get_analytics_charts(
        self, 
        session_id: Optional[str] = None, 
        time_range: str = "30d"
    ) -> Dict[str, Any]:
        """Get analytics data for charts"""
        
        # Generate mock analytics data
        if time_range == "30d":
            days = 30
        elif time_range == "7d":
            days = 7
        elif time_range == "90d":
            days = 90
        else:
            days = 30
        
        # Generate time series data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        time_series_data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Mock fouling density data
            fouling_density = 20 + np.random.normal(0, 5) + (current_date - start_date).days * 0.5
            fouling_density = max(0, min(100, fouling_density))
            
            time_series_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "fouling_density": round(fouling_density, 2),
                "fuel_cost": round(fouling_density * 2.5, 2),
                "maintenance_cost": round(fouling_density * 15, 2)
            })
            
            current_date += timedelta(days=1)
        
        # Species distribution data
        species_distribution = [
            {"species": "Barnacles", "coverage": 35.2, "count": 12},
            {"species": "Seaweed", "coverage": 28.7, "count": 8},
            {"species": "Mussels", "coverage": 18.3, "count": 5},
            {"species": "Sponges", "coverage": 12.1, "count": 3},
            {"species": "Anemones", "coverage": 5.7, "count": 2}
        ]
        
        # Cost vs delay projection
        delay_days = list(range(0, 91, 7))  # 0 to 90 days, weekly intervals
        cost_projection = [
            {
                "delay_days": days,
                "cleaning_cost": 1000 + days * 50,  # Base cost + daily increase
                "fuel_cost": days * 25,  # Daily fuel cost increase
                "total_cost": 1000 + days * 75
            }
            for days in delay_days
        ]
        
        return {
            "time_series": time_series_data,
            "species_distribution": species_distribution,
            "cost_projection": cost_projection,
            "summary": {
                "total_sessions": 15,
                "avg_coverage": 32.5,
                "dominant_species": "Barnacles",
                "total_cost_saved": 12500
            }
        }
    
    async def save_analytics(self, session_id: str, analytics_data: Dict[str, Any]):
        """Save analytics data for a session"""
        analytics = self._load_json(self.analytics_file)
        analytics[session_id] = {
            **analytics_data,
            "timestamp": datetime.now().isoformat()
        }
        self._save_json(self.analytics_file, analytics)
    
    async def get_maintenance_schedule(self) -> List[Dict[str, Any]]:
        """Get maintenance schedule recommendations"""
        return [
            {
                "id": 1,
                "title": "Hull Cleaning - Port Side",
                "description": "High barnacle density detected in port side mid-hull region",
                "priority": "high",
                "estimated_cost": 2500,
                "estimated_duration": "4 hours",
                "recommended_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "species": ["Barnacles", "Seaweed"],
                "coverage": 45.2
            },
            {
                "id": 2,
                "title": "Propeller Maintenance",
                "description": "Moderate fouling on propeller blades affecting efficiency",
                "priority": "medium",
                "estimated_cost": 1800,
                "estimated_duration": "2 hours",
                "recommended_date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                "species": ["Mussels", "Seaweed"],
                "coverage": 28.7
            },
            {
                "id": 3,
                "title": "Routine Hull Inspection",
                "description": "Quarterly routine inspection and minor cleaning",
                "priority": "low",
                "estimated_cost": 1200,
                "estimated_duration": "6 hours",
                "recommended_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                "species": ["Sponges", "Anemones"],
                "coverage": 12.1
            }
        ]
    
    async def add_maintenance_task(self, task_data: Dict[str, Any]):
        """Add a new maintenance task"""
        maintenance_file = self.data_dir / "maintenance.json"
        tasks = self._load_json(maintenance_file)
        
        task_id = len(tasks) + 1
        tasks[task_id] = {
            "id": task_id,
            **task_data,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        self._save_json(maintenance_file, tasks)
        return task_id
