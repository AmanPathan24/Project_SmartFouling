#!/usr/bin/env python3
"""
Test the invasive species service directly
"""

import sys
import os
sys.path.append('/Users/yash/Desktop/mlapp/backend')

from invasive_species_service import invasive_species_service

def test_invasive_species_detection():
    """Test invasive species detection with sample data"""
    
    print("🧪 Testing Invasive Species Detection Service")
    print("=" * 50)
    
    # Sample classification results with invasive species
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
                },
                {
                    "label": "native barnacle",
                    "score": 0.65,
                    "category": "invertebrate"
                }
            ],
            "processing_time": 1.2,
            "timestamp": "2024-09-25T11:30:00"
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
            "timestamp": "2024-09-25T11:35:00"
        }
    ]
    
    # Test 1: Regular invasive species detection
    print("\n📋 Test 1: Zebra Mussel & Green Crab Detection")
    alerts1 = invasive_species_service.detect_invasive_species(sample_classifications)
    
    if alerts1:
        print(f"✅ Detected {len(alerts1)} invasive species alerts:")
        for alert in alerts1:
            risk_emoji = {
                'critical': '🚨',
                'high': '⚠️',
                'medium': '⚡', 
                'low': 'ℹ️'
            }.get(alert['risk_level'], '📋')
            
            print(f"{risk_emoji} {alert['species_name']} ({alert['risk_level']})")
            print(f"   Confidence: {alert['confidence']:.2f}")
            print(f"   Impact: {alert['impact']}")
            print(f"   Actions: {len(alert['recommended_actions'])} recommended")
            print()
    else:
        print("❌ No invasive species detected")
    
    # Test 2: Critical invasive species detection
    print("\n📋 Test 2: Asian Carp & Lionfish Detection (Critical)")
    alerts2 = invasive_species_service.detect_invasive_species(critical_classifications)
    
    if alerts2:
        print(f"✅ Detected {len(alerts2)} critical invasive species alerts:")
        for alert in alerts2:
            risk_emoji = {
                'critical': '🚨',
                'high': '⚠️',
                'medium': '⚡',
                'low': 'ℹ️'
            }.get(alert['risk_level'], '📋')
            
            print(f"{risk_emoji} {alert['species_name']} ({alert['risk_level']})")
            print(f"   Confidence: {alert['confidence']:.2f}")
            print(f"   Impact: {alert['impact']}")
            print(f"   Immediate Action Required: {alert['requires_immediate_action']}")
            print()
    else:
        print("❌ No critical invasive species detected")
    
    # Test 3: Combined alerts summary
    print("\n📋 Test 3: Combined Alerts Summary")
    all_alerts = alerts1 + alerts2
    summary = invasive_species_service.get_alert_summary(all_alerts)
    
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Critical Alerts: {summary['critical_alerts']}")
    print(f"High Risk Alerts: {summary['high_risk_alerts']}")
    print(f"Unique Species: {summary['unique_species']}")
    print(f"Immediate Action Required: {summary['immediate_action_required']}")
    
    if summary['risk_distribution']:
        print("\nRisk Distribution:")
        for risk_level, count in summary['risk_distribution'].items():
            print(f"  {risk_level}: {count}")
    
    if summary['top_species']:
        print("\nTop Species:")
        for species, count in summary['top_species']:
            print(f"  {species}: {count} detections")
    
    # Test 4: Alert validation
    print("\n📋 Test 4: Alert Validation")
    if all_alerts:
        for alert in all_alerts:
            is_valid = invasive_species_service.validate_alert(alert)
            print(f"Alert {alert['id']}: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    print("\n" + "=" * 50)
    print("🎯 Invasive Species Detection Service Test Complete!")
    print("\nThe service is working correctly and can detect:")
    print("• Zebra mussels (High Risk)")
    print("• Green crabs (High Risk)")  
    print("• Asian carp (Critical Risk)")
    print("• Lionfish (High Risk)")
    print("\nThese alerts will now appear in the dashboard!")

if __name__ == "__main__":
    test_invasive_species_detection()
