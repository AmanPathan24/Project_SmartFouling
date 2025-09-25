"""
Invasive Species Detection Service
Monitors classification results and identifies invasive marine species
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class InvasiveSpeciesService:
    """Service for detecting invasive marine species from classification results"""
    
    def __init__(self):
        # Define invasive species database with risk levels and regions
        self.invasive_species_db = {
            # High-risk invasive species
            "zebra_mussel": {
                "scientific_name": "Dreissena polymorpha",
                "risk_level": "critical",
                "impact": "Blocks water intake pipes, outcompetes native species",
                "regions": ["freshwater", "brackish"],
                "detection_keywords": ["zebra", "mussel", "dreissena", "polymorpha"],
                "confidence_threshold": 0.7
            },
            "green_crab": {
                "scientific_name": "Carcinus maenas",
                "risk_level": "high",
                "impact": "Predates on native shellfish, alters ecosystem structure",
                "regions": ["coastal", "estuarine"],
                "detection_keywords": ["green", "crab", "carcinus", "maenas"],
                "confidence_threshold": 0.6
            },
            "asian_carp": {
                "scientific_name": "Hypophthalmichthys spp.",
                "risk_level": "critical",
                "impact": "Outcompetes native fish, disrupts food chains",
                "regions": ["freshwater", "riverine"],
                "detection_keywords": ["asian", "carp", "hypophthalmichthys", "silver", "bighead"],
                "confidence_threshold": 0.8
            },
            "lionfish": {
                "scientific_name": "Pterois volitans",
                "risk_level": "high",
                "impact": "Venomous predator, reduces native fish populations",
                "regions": ["marine", "coral_reef"],
                "detection_keywords": ["lionfish", "pterois", "volitans", "lion", "fish"],
                "confidence_threshold": 0.7
            },
            "caulerpa_taxifolia": {
                "scientific_name": "Caulerpa taxifolia",
                "risk_level": "high",
                "impact": "Forms dense mats, smothers native vegetation",
                "regions": ["marine", "subtidal"],
                "detection_keywords": ["caulerpa", "taxifolia", "killer", "algae", "seaweed"],
                "confidence_threshold": 0.6
            },
            "codium_fragile": {
                "scientific_name": "Codium fragile",
                "risk_level": "medium",
                "impact": "Displaces native seaweed, alters habitat structure",
                "regions": ["marine", "rocky_intertidal"],
                "detection_keywords": ["codium", "fragile", "dead", "mans", "fingers"],
                "confidence_threshold": 0.5
            },
            "mnemiopsis_leidyi": {
                "scientific_name": "Mnemiopsis leidyi",
                "risk_level": "critical",
                "impact": "Comb jelly that devastates fish populations",
                "regions": ["marine", "estuarine"],
                "detection_keywords": ["mnemiopsis", "leidyi", "comb", "jelly", "ctenophore"],
                "confidence_threshold": 0.8
            },
            "dreissena_bugensis": {
                "scientific_name": "Dreissena bugensis",
                "risk_level": "high",
                "impact": "Quagga mussel, blocks infrastructure, outcompetes natives",
                "regions": ["freshwater", "lake"],
                "detection_keywords": ["quagga", "mussel", "dreissena", "bugensis"],
                "confidence_threshold": 0.7
            }
        }
        
        # Risk level priorities
        self.risk_priorities = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }

    def detect_invasive_species(self, classification_results: List[Dict]) -> List[Dict]:
        """
        Analyze classification results for invasive species
        
        Args:
            classification_results: List of classification results from API
            
        Returns:
            List of invasive species alerts
        """
        alerts = []
        
        for result in classification_results:
            if not result.get('success', False):
                continue
                
            classifications = result.get('classifications', [])
            if not classifications:
                continue
            
            for classification in classifications:
                species_name = classification.get('label', '').lower()
                confidence = classification.get('score', 0.0)
                
                # Check against invasive species database
                detected_species = self._match_invasive_species(species_name, confidence)
                
                if detected_species:
                    alert = self._create_alert(detected_species, confidence, result)
                    alerts.append(alert)
        
        # Sort alerts by risk level (highest first)
        alerts.sort(key=lambda x: self.risk_priorities.get(x['risk_level'], 0), reverse=True)
        
        return alerts

    def _match_invasive_species(self, species_name: str, confidence: float) -> Optional[Dict]:
        """Match species name against invasive species database"""
        
        for species_key, species_data in self.invasive_species_db.items():
            # Check if any detection keywords match
            for keyword in species_data['detection_keywords']:
                if keyword.lower() in species_name:
                    # Check confidence threshold
                    if confidence >= species_data['confidence_threshold']:
                        return {
                            'key': species_key,
                            'data': species_data,
                            'matched_keyword': keyword,
                            'original_name': species_name
                        }
        
        return None

    def _create_alert(self, detected_species: Dict, confidence: float, classification_result: Dict) -> Dict:
        """Create an invasive species alert"""
        
        species_data = detected_species['data']
        
        alert = {
            'id': f"invasive_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{detected_species['key']}",
            'type': 'invasive_species',
            'risk_level': species_data['risk_level'],
            'priority': self.risk_priorities[species_data['risk_level']],
            'species_name': species_data['scientific_name'],
            'common_name': detected_species['key'].replace('_', ' ').title(),
            'confidence': confidence,
            'detected_keyword': detected_species['matched_keyword'],
            'original_classification': detected_species['original_name'],
            'impact': species_data['impact'],
            'affected_regions': species_data['regions'],
            'detection_time': datetime.now().isoformat(),
            'classification_result': classification_result,
            'recommended_actions': self._get_recommended_actions(species_data['risk_level']),
            'status': 'active',
            'requires_immediate_action': species_data['risk_level'] in ['critical', 'high']
        }
        
        return alert

    def _get_recommended_actions(self, risk_level: str) -> List[str]:
        """Get recommended actions based on risk level"""
        
        actions = {
            'critical': [
                "Immediate containment measures required",
                "Notify marine authorities within 24 hours",
                "Implement emergency response protocol",
                "Document with photographs and GPS coordinates",
                "Begin eradication procedures if feasible"
            ],
            'high': [
                "Report to local marine authorities within 48 hours",
                "Implement monitoring and tracking protocol",
                "Consider containment measures",
                "Document species and location details",
                "Assess potential for local eradication"
            ],
            'medium': [
                "Report to marine monitoring database",
                "Increase monitoring frequency in affected area",
                "Document and track population changes",
                "Assess ecological impact",
                "Consider management strategies"
            ],
            'low': [
                "Document in species monitoring log",
                "Continue regular monitoring",
                "Track population trends",
                "Assess potential risks"
            ]
        }
        
        return actions.get(risk_level, ["Monitor and document"])

    def get_alert_summary(self, alerts: List[Dict]) -> Dict:
        """Generate summary statistics for invasive species alerts"""
        
        if not alerts:
            return {
                'total_alerts': 0,
                'critical_alerts': 0,
                'high_risk_alerts': 0,
                'unique_species': 0,
                'immediate_action_required': False
            }
        
        summary = {
            'total_alerts': len(alerts),
            'critical_alerts': len([a for a in alerts if a['risk_level'] == 'critical']),
            'high_risk_alerts': len([a for a in alerts if a['risk_level'] == 'high']),
            'unique_species': len(set(a['species_name'] for a in alerts)),
            'immediate_action_required': any(a['requires_immediate_action'] for a in alerts),
            'risk_distribution': {},
            'top_species': []
        }
        
        # Risk level distribution
        for alert in alerts:
            risk_level = alert['risk_level']
            summary['risk_distribution'][risk_level] = summary['risk_distribution'].get(risk_level, 0) + 1
        
        # Top species by frequency
        species_counts = {}
        for alert in alerts:
            species = alert['species_name']
            species_counts[species] = species_counts.get(species, 0) + 1
        
        summary['top_species'] = sorted(
            species_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return summary

    def validate_alert(self, alert: Dict) -> bool:
        """Validate alert data integrity"""
        
        required_fields = [
            'id', 'type', 'risk_level', 'species_name', 
            'confidence', 'detection_time', 'status'
        ]
        
        for field in required_fields:
            if field not in alert:
                logger.error(f"Alert missing required field: {field}")
                return False
        
        if alert['risk_level'] not in self.risk_priorities:
            logger.error(f"Invalid risk level: {alert['risk_level']}")
            return False
        
        if not (0 <= alert['confidence'] <= 1):
            logger.error(f"Invalid confidence score: {alert['confidence']}")
            return False
        
        return True

# Global instance
invasive_species_service = InvasiveSpeciesService()
