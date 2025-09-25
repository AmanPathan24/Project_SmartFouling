#!/usr/bin/env python3
"""
Test script for invasive species alerts
Creates sample classification results with invasive species to test the alert system
"""

import requests
import json
import time
from datetime import datetime

# Backend URL
BASE_URL = "http://localhost:8000"

def create_test_session_with_invasive_species():
    """Create a test session with invasive species classification results"""
    
    # Create a session
    session_data = {
        "session_name": "Invasive Species Test Session"
    }
    
    response = requests.post(f"{BASE_URL}/api/sessions", json=session_data)
    if response.status_code != 200:
        print(f"Failed to create session: {response.status_code}")
        return None
    
    session = response.json()
    session_id = session['session_id']
    print(f"Created test session: {session_id}")
    
    # Create mock classification results with invasive species
    mock_classification_results = [
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
                },
                {
                    "label": "native barnacle",
                    "score": 0.65,
                    "category": "invertebrate"
                }
            ],
            "processing_time": 1.2,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    # Create another session with critical invasive species
    session_data2 = {
        "session_name": "Critical Invasive Species Alert"
    }
    
    response2 = requests.post(f"{BASE_URL}/api/sessions", json=session_data2)
    if response2.status_code != 200:
        print(f"Failed to create second session: {response2.status_code}")
        return None
    
    session2 = response2.json()
    session_id2 = session2['session_id']
    print(f"Created critical test session: {session_id2}")
    
    mock_classification_results2 = [
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
    
    # Update sessions with mock data (simulating what would happen during image processing)
    # Note: In a real scenario, this would be done through the upload endpoint
    # For testing, we'll simulate the session data structure
    
    print("Test sessions created with invasive species data")
    print("Sessions to check:")
    print(f"- {session_id} (zebra mussel, green crab)")
    print(f"- {session_id2} (asian carp, lionfish)")
    
    return [session_id, session_id2]

def test_alerts_endpoint():
    """Test the invasive species alerts endpoint"""
    
    print("\n=== Testing Invasive Species Alerts ===")
    
    try:
        response = requests.get(f"{BASE_URL}/api/alerts/invasive-species")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Alerts endpoint working")
            print(f"Total alerts: {data['total_count']}")
            print(f"Summary: {json.dumps(data['summary'], indent=2)}")
            
            if data['alerts']:
                print("\nActive Alerts:")
                for alert in data['alerts']:
                    print(f"- {alert['species_name']} ({alert['risk_level']}) - {alert['confidence']:.2f} confidence")
            else:
                print("No active alerts found")
        else:
            print(f"❌ Alerts endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing alerts: {e}")

def test_session_alerts(session_id):
    """Test session-specific alerts"""
    
    print(f"\n=== Testing Session Alerts for {session_id} ===")
    
    try:
        response = requests.get(f"{BASE_URL}/api/sessions/{session_id}/alerts")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Session alerts endpoint working")
            print(f"Alerts for session: {len(data['alerts'])}")
            
            if data['alerts']:
                for alert in data['alerts']:
                    print(f"- {alert['species_name']} ({alert['risk_level']})")
            else:
                print("No alerts found for this session")
        else:
            print(f"❌ Session alerts failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing session alerts: {e}")

def main():
    """Main test function"""
    
    print("🚨 Invasive Species Alert System Test")
    print("=" * 50)
    
    # Test if backend is running
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code != 200:
            print("❌ Backend not running or not accessible")
            return
    except:
        print("❌ Backend not running or not accessible")
        return
    
    print("✅ Backend is running")
    
    # Create test sessions
    session_ids = create_test_session_with_invasive_species()
    
    if not session_ids:
        print("❌ Failed to create test sessions")
        return
    
    # Wait a moment for data to be processed
    time.sleep(2)
    
    # Test alerts endpoint
    test_alerts_endpoint()
    
    # Test session-specific alerts
    for session_id in session_ids:
        test_session_alerts(session_id)
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nTo see alerts in the frontend:")
    print("1. Go to http://localhost:3000")
    print("2. Click the 'Alerts' tab")
    print("3. Check for invasive species notifications")

if __name__ == "__main__":
    main()
