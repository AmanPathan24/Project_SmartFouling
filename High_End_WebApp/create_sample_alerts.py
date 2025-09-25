#!/usr/bin/env python3
"""
Create sample invasive species alerts by directly updating session data
"""

import requests
import json
import os
from datetime import datetime

# Backend URL
BASE_URL = "http://localhost:8000"

def create_sample_alerts():
    """Create sample alerts by directly modifying session files"""
    
    # Sample invasive species classification results
    sample_classifications = [
        {
            "success": True,
            "model": "resnet50",
            "classifications": [
                {
                    "label": "zebra mussel",
                    "score": 0.85,
                    "category": "invertebrate"
                },
                {
                    "label": "green crab",
                    "score": 0.72,
                    "category": "crustacean"
                }
            ],
            "processing_time": 1.2,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    critical_classifications = [
        {
            "success": True,
            "model": "resnet50", 
            "classifications": [
                {
                    "label": "asian carp",
                    "score": 0.92,
                    "category": "fish"
                },
                {
                    "label": "lionfish",
                    "score": 0.88,
                    "category": "fish"
                }
            ],
            "processing_time": 1.5,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    # Create sessions
    session1_data = {"session_name": "Invasive Species Test - Zebra Mussel & Green Crab"}
    session2_data = {"session_name": "CRITICAL ALERT - Asian Carp & Lionfish"}
    
    response1 = requests.post(f"{BASE_URL}/api/sessions", json=session1_data)
    response2 = requests.post(f"{BASE_URL}/api/sessions", json=session2_data)
    
    if response1.status_code != 200 or response2.status_code != 200:
        print("Failed to create test sessions")
        return
    
    session1 = response1.json()
    session2 = response2.json()
    
    print(f"Created sessions:")
    print(f"- {session1['session_id']}: {session1['session_name']}")
    print(f"- {session2['session_id']}: {session2['session_name']}")
    
    # Manually add classification results to session files
    # This simulates what would happen during image processing
    
    # Get session data and add classification results
    try:
        # Update session 1
        session1_response = requests.get(f"{BASE_URL}/api/sessions/{session1['session_id']}")
        if session1_response.status_code == 200:
            session1_data = session1_response.json()
            session1_data['classification_results'] = sample_classifications
            session1_data['analysis_results'] = [
                {
                    "filename": "test_invasive_1.jpg",
                    "confidence": 0.85,
                    "dominant_class": "zebra_mussel",
                    "api_classification": sample_classifications[0]
                }
            ]
            
            # Write to session file directly
            session_file = f"backend/sessions/{session1['session_id']}.json"
            with open(session_file, 'w') as f:
                json.dump(session1_data, f, indent=2)
            print(f"✅ Updated session 1 with classification data")
        
        # Update session 2  
        session2_response = requests.get(f"{BASE_URL}/api/sessions/{session2['session_id']}")
        if session2_response.status_code == 200:
            session2_data = session2_response.json()
            session2_data['classification_results'] = critical_classifications
            session2_data['analysis_results'] = [
                {
                    "filename": "test_critical_1.jpg", 
                    "confidence": 0.92,
                    "dominant_class": "asian_carp",
                    "api_classification": critical_classifications[0]
                }
            ]
            
            # Write to session file directly
            session_file = f"backend/sessions/{session2['session_id']}.json"
            with open(session_file, 'w') as f:
                json.dump(session2_data, f, indent=2)
            print(f"✅ Updated session 2 with critical classification data")
            
    except Exception as e:
        print(f"Error updating sessions: {e}")
        return
    
    print("\n🎯 Sample invasive species data created!")
    print("Check the alerts now:")
    print("1. Go to http://localhost:3000")
    print("2. Click the 'Alerts' tab")
    print("3. You should see invasive species alerts")

def test_alerts():
    """Test the alerts after creating sample data"""
    
    print("\n=== Testing Alerts ===")
    
    try:
        response = requests.get(f"{BASE_URL}/api/alerts/invasive-species")
        if response.status_code == 200:
            data = response.json()
            print(f"Total alerts: {data['total_count']}")
            
            if data['alerts']:
                print("\n🚨 Active Invasive Species Alerts:")
                for alert in data['alerts']:
                    risk_emoji = {
                        'critical': '🚨',
                        'high': '⚠️', 
                        'medium': '⚡',
                        'low': 'ℹ️'
                    }.get(alert['risk_level'], '📋')
                    
                    print(f"{risk_emoji} {alert['species_name']} ({alert['risk_level']}) - {alert['confidence']:.2f} confidence")
                    print(f"   Impact: {alert['impact']}")
                    print(f"   Session: {alert.get('session_name', 'Unknown')}")
                    print()
            else:
                print("No alerts found")
        else:
            print(f"Failed to get alerts: {response.status_code}")
            
    except Exception as e:
        print(f"Error testing alerts: {e}")

if __name__ == "__main__":
    create_sample_alerts()
    test_alerts()
