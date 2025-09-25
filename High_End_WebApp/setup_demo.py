#!/usr/bin/env python3
"""
Demo Data Setup Script for Marine Biofouling Detection System
Creates sample data and demo sessions for testing
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

def create_demo_data():
    """Create demo data for the application"""
    print("🎭 Setting up demo data...")
    
    # Create backend data directory
    backend_data_dir = Path("backend/data")
    backend_data_dir.mkdir(exist_ok=True)
    
    # Create demo sessions
    demo_sessions = {
        "session_20241201_143022_abc123": {
            "session_id": "session_20241201_143022_abc123",
            "session_name": "Hull Inspection - Port Side",
            "model_name": "biofouling-detector-v1",
            "confidence_threshold": 0.6,
            "status": "completed",
            "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
            "completed_at": (datetime.now() - timedelta(days=2, hours=1)).isoformat(),
            "images": [
                {
                    "filename": "hull1_original.jpg",
                    "original_path": "uploads/session_20241201_143022_abc123_hull1_original.jpg",
                    "preprocessed_path": "processed/session_20241201_143022_abc123_hull1_original.jpg",
                    "segmentation_path": "outputs/session_20241201_143022_abc123_mask_hull1_original.jpg",
                    "source": "manual",
                    "uploaded_at": (datetime.now() - timedelta(days=2)).isoformat(),
                    "size": 2048576,
                    "dimensions": [1920, 1080]
                }
            ],
            "analysis_results": [
                {
                    "image_id": "hull1_original.jpg",
                    "detections": [
                        {
                            "species": "Barnacles",
                            "scientific_name": "Balanus spp.",
                            "confidence": 0.87,
                            "coverage_percentage": 23.5,
                            "bbox": {"x": 150, "y": 200, "width": 300, "height": 250}
                        },
                        {
                            "species": "Seaweed",
                            "scientific_name": "Various algae",
                            "confidence": 0.72,
                            "coverage_percentage": 15.2,
                            "bbox": {"x": 400, "y": 300, "width": 200, "height": 180}
                        }
                    ],
                    "total_coverage": 38.7,
                    "dominant_species": "Barnacles",
                    "processing_time": 2.3
                }
            ],
            "analytics": {
                "total_coverage": 38.7,
                "species_count": 2,
                "dominant_species": "Barnacles",
                "avg_confidence": 0.795,
                "total_detections": 2,
                "fuel_cost_impact": 96.75,
                "maintenance_cost": 580.5,
                "cleaning_urgency": "medium",
                "species_distribution": {
                    "Barnacles": 23.5,
                    "Seaweed": 15.2
                }
            }
        },
        "session_20241202_091545_def456": {
            "session_id": "session_20241202_091545_def456",
            "session_name": "Propeller Inspection",
            "model_name": "biofouling-detector-v1",
            "confidence_threshold": 0.5,
            "status": "completed",
            "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
            "completed_at": (datetime.now() - timedelta(days=1, hours=2)).isoformat(),
            "images": [
                {
                    "filename": "prop_original.jpg",
                    "original_path": "uploads/session_20241202_091545_def456_prop_original.jpg",
                    "preprocessed_path": "processed/session_20241202_091545_def456_prop_original.jpg",
                    "segmentation_path": "outputs/session_20241202_091545_def456_mask_prop_original.jpg",
                    "source": "dataset",
                    "uploaded_at": (datetime.now() - timedelta(days=1)).isoformat(),
                    "size": 1536000,
                    "dimensions": [1280, 720]
                }
            ],
            "analysis_results": [
                {
                    "image_id": "prop_original.jpg",
                    "detections": [
                        {
                            "species": "Mussels",
                            "scientific_name": "Mytilus spp.",
                            "confidence": 0.91,
                            "coverage_percentage": 31.8,
                            "bbox": {"x": 200, "y": 150, "width": 400, "height": 300}
                        }
                    ],
                    "total_coverage": 31.8,
                    "dominant_species": "Mussels",
                    "processing_time": 1.8
                }
            ],
            "analytics": {
                "total_coverage": 31.8,
                "species_count": 1,
                "dominant_species": "Mussels",
                "avg_confidence": 0.91,
                "total_detections": 1,
                "fuel_cost_impact": 79.5,
                "maintenance_cost": 477,
                "cleaning_urgency": "medium",
                "species_distribution": {
                    "Mussels": 31.8
                }
            }
        }
    }
    
    # Save sessions
    sessions_file = backend_data_dir / "sessions.json"
    with open(sessions_file, 'w') as f:
        json.dump(demo_sessions, f, indent=2)
    
    print(f"✅ Created {len(demo_sessions)} demo sessions")
    
    # Create demo datasets
    demo_datasets = {
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
        }
    }
    
    datasets_file = backend_data_dir / "datasets.json"
    with open(datasets_file, 'w') as f:
        json.dump(demo_datasets, f, indent=2)
    
    print(f"✅ Created {len(demo_datasets)} demo datasets")
    
    # Create demo analytics data
    demo_analytics = {}
    analytics_file = backend_data_dir / "analytics.json"
    with open(analytics_file, 'w') as f:
        json.dump(demo_analytics, f, indent=2)
    
    print("✅ Created analytics data structure")
    
    # Copy demo images if they exist
    demo_source_dir = Path("webapp/public/demo")
    demo_dest_dir = Path("backend/static/demo")
    
    if demo_source_dir.exists():
        demo_dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy demo images
        for image_file in demo_source_dir.glob("*.jpg"):
            dest_file = demo_dest_dir / image_file.name
            if not dest_file.exists():
                import shutil
                shutil.copy2(image_file, dest_file)
                print(f"✅ Copied demo image: {image_file.name}")
    
    print("\n🎉 Demo data setup completed!")
    print("\n📋 Available demo sessions:")
    for session_id, session_data in demo_sessions.items():
        print(f"   • {session_data['session_name']} ({session_data['status']})")
    
    print("\n📂 Available datasets:")
    for dataset_name, dataset_data in demo_datasets.items():
        print(f"   • {dataset_data['name']} ({len(dataset_data['images'])} images)")

def main():
    """Main function"""
    print("🚢 Marine Biofouling Detection System - Demo Setup")
    print("=" * 50)
    
    try:
        create_demo_data()
        print("\n✅ Demo setup completed successfully!")
        print("\n🌐 You can now start the application with:")
        print("   ./start.sh")
        
    except Exception as e:
        print(f"\n❌ Demo setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
