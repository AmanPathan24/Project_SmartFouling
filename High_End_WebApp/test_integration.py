#!/usr/bin/env python3
"""
Integration Test Script for Marine Biofouling Detection System
Tests the complete end-to-end workflow
"""

import requests
import time
import json
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

def test_backend_health():
    """Test backend health endpoint"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend connection failed: {e}")
        return False

def test_frontend_access():
    """Test frontend accessibility"""
    print("🔍 Testing frontend access...")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend access failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend connection failed: {e}")
        return False

def test_api_endpoints():
    """Test key API endpoints"""
    print("🔍 Testing API endpoints...")
    
    endpoints = [
        ("/api/health", "GET"),
        ("/api/datasets", "GET"),
        ("/api/sessions", "GET"),
    ]
    
    results = []
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=10)
            
            if response.status_code in [200, 404]:  # 404 is OK for empty datasets
                print(f"✅ {method} {endpoint} - {response.status_code}")
                results.append(True)
            else:
                print(f"❌ {method} {endpoint} - {response.status_code}")
                results.append(False)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {method} {endpoint} - Connection error: {e}")
            results.append(False)
    
    return all(results)

def test_session_workflow():
    """Test complete session workflow"""
    print("🔍 Testing session workflow...")
    
    try:
        # Create a session
        session_data = {
            "session_name": "Test Session",
            "model_name": "biofouling-detector-v1",
            "confidence_threshold": 0.5
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/sessions",
            data=session_data,
            timeout=10
        )
        
        if response.status_code == 200:
            session_id = response.json()["session_id"]
            print(f"✅ Session created: {session_id}")
            
            # Test session retrieval
            get_response = requests.get(f"{BACKEND_URL}/api/sessions/{session_id}", timeout=10)
            if get_response.status_code == 200:
                print("✅ Session retrieved successfully")
                return True
            else:
                print(f"❌ Session retrieval failed: {get_response.status_code}")
                return False
        else:
            print(f"❌ Session creation failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Session workflow test failed: {e}")
        return False

def test_file_structure():
    """Test file structure and dependencies"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "backend/main.py",
        "backend/preprocessing_service.py",
        "backend/model_service.py",
        "backend/database.py",
        "backend/utils.py",
        "backend/requirements.txt",
        "frontend/package.json",
        "frontend/src/App.tsx",
        "frontend/src/main.tsx",
        "start.sh"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all integration tests"""
    print("🚢 Marine Biofouling Detection System - Integration Tests")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Backend Health", test_backend_health),
        ("Frontend Access", test_frontend_access),
        ("API Endpoints", test_api_endpoints),
        ("Session Workflow", test_session_workflow),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        print("\n🌐 Access your application:")
        print(f"   Frontend: {FRONTEND_URL}")
        print(f"   Backend:  {BACKEND_URL}")
        print(f"   API Docs: {BACKEND_URL}/docs")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
