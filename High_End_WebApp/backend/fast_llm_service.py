"""
Fast LLM Service for Marine Biofouling Report Generation
Optimized for speed and reliability with instant fallback
"""

import os
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def generate_instant_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate instant report without external API calls"""
    try:
        analytics = data.get("analytics", {})
        
        # Extract key data points
        coverage = analytics.get('total_coverage', 0)
        species = analytics.get('dominant_species', 'Unknown')
        confidence = analytics.get('avg_confidence', 0)
        urgency = analytics.get('cleaning_urgency', 'Medium')
        fuel_cost = analytics.get('fuel_cost_impact', 0)
        maintenance_cost = analytics.get('maintenance_cost', 0)
        
        # Generate comprehensive report instantly
        report_content = f"""
{"="*80}
        COMPREHENSIVE MARINE BIOFOULING ASSESSMENT REPORT
{"="*80}

**Date of Analysis:** {datetime.now().strftime("%A, %B %d, %Y")}
**Session:** {data.get('session_name', 'Unknown')}
**Location:** Marine Biofouling Detection System

{"-" * 80}

**Input Data Points for this Analysis:**
- Overall Surface Density: {coverage:.1f}%
- Dominant Species: {species}
- Average Confidence: {confidence:.2f}
- Cleaning Urgency: {urgency}
- Fuel Cost Impact: ${fuel_cost:.2f}
- Maintenance Cost: ${maintenance_cost:.2f}

{"-" * 80}

**AI-Generated Analysis & Recommendations:**

**1. Executive Summary:**
Analysis of session {data.get('session_name', 'Unknown')} reveals {coverage:.1f}% fouling coverage with {confidence:.2f} confidence level. The dominant species is {species}, indicating {urgency.lower()} priority cleaning requirements.

**2. Detailed Analysis:**
The fouling levels suggest {'high' if coverage > 15 else 'moderate' if coverage > 5 else 'low'} biofouling accumulation. Surface coverage of {coverage:.1f}% indicates {'significant' if coverage > 10 else 'moderate' if coverage > 5 else 'minimal'} marine growth that {'could significantly impact' if coverage > 15 else 'may impact' if coverage > 5 else 'has minimal impact on'} vessel performance.

**3. Risk Assessment:**
Current fouling levels pose a {urgency.lower()} risk to vessel performance. Increased drag from biofouling can lead to:
- {5 + (coverage * 0.5):.1f}% increase in fuel consumption
- Reduced maneuverability and speed
- Potential corrosion and hull damage
- Risk of invasive species transfer
- Estimated annual fuel cost impact: ${fuel_cost * 12:.2f}

**4. Recommended Mitigation Strategies:**
1. **Immediate Cleaning**: {'Schedule emergency hull cleaning within 3-7 days' if coverage > 15 else 'Schedule hull cleaning within 7-14 days' if coverage > 5 else 'Schedule routine cleaning within 30 days'}
2. **Anti-fouling Coating**: Apply copper-based anti-fouling paint after cleaning
3. **Preventive Measures**: Implement regular inspection schedule every 3 months
4. **Monitoring**: Use advanced imaging systems for continuous monitoring

**5. Maintenance Schedule:**
- **Immediate Action**: {'Emergency cleaning within 3 days' if coverage > 15 else 'Cleaning within 7 days' if coverage > 5 else 'Routine cleaning within 30 days'} (High Priority)
- **Secondary Action**: Anti-fouling coating application within 30 days (Medium Priority)
- **Follow-up**: Next inspection in {'60 days' if coverage > 10 else '90 days'} (Low Priority)

**6. Cost-Benefit Analysis:**
- Immediate cleaning cost: ${maintenance_cost:.2f}
- Annual fuel savings potential: ${fuel_cost * 12:.2f}
- ROI: {((fuel_cost * 12) / maintenance_cost * 100):.1f}% return on investment
- Break-even period: {(maintenance_cost / (fuel_cost * 12) * 12):.1f} months

**7. Action Items:**
1. Contact cleaning service provider immediately
2. Schedule dry dock availability
3. Order anti-fouling materials
4. Update maintenance records and schedule
5. Implement monitoring protocol

{"-" * 80}
                 END OF REPORT
{"="*80}
"""
        
        return {
            "analysis": report_content,
            "timestamp": datetime.now().isoformat(),
            "model_used": "instant_analysis_engine",
            "raw_content": report_content
        }
        
    except Exception as e:
        logger.error(f"Failed to generate instant report: {e}")
        return {
            "analysis": f"Error generating report: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "model_used": "error"
        }
