"""
LLM Service for Marine Biofouling Report Generation
Uses Hugging Face OpenAI-compatible API
"""

import os
from datetime import datetime
from typing import Dict, Any
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def format_report_content(raw_report_text: str, fouling_data: Dict[str, str], session_data: Dict[str, Any]) -> str:
    """
    Format the raw LLM response into a well-structured report
    """
    formatted_report = f"""
{"="*80}
        COMPREHENSIVE MARINE BIOFOULING ASSESSMENT REPORT
{"="*80}

**Date of Analysis:** {datetime.now().strftime("%A, %B %d, %Y")}
**Session:** {session_data.get('session_name', 'Unknown')}
**Location:** Marine Biofouling Detection System

{"-" * 80}

**Input Data Points for this Analysis:**
"""
    
    for key, value in fouling_data.items():
        formatted_report += f"- {key}: {value}\n"
    
    formatted_report += f"""
{"-" * 80}

**AI-Generated Analysis & Recommendations:**

{raw_report_text}

{"-" * 80}
                 END OF REPORT
{"="*80}
"""
    
    return formatted_report

def generate_fallback_report(fouling_data: Dict[str, str], session_data: Dict[str, Any]) -> str:
    """Generate a fallback report when LLM API fails"""
    coverage = fouling_data.get('Overall Surface Density', '0%')
    species = fouling_data.get('Dominant Species', 'Unknown')
    urgency = fouling_data.get('Cleaning Urgency', 'Medium')
    
    return f"""
**Executive Summary:**
Analysis of session {session_data.get('session_name', 'Unknown')} reveals {coverage} fouling coverage. The dominant species detected is {species}, indicating {urgency.lower()} priority cleaning requirements.

**Detailed Analysis:**
The fouling levels suggest moderate to high biofouling accumulation requiring attention. Surface coverage of {coverage} indicates significant marine growth that could impact vessel performance.

**Risk Assessment:**
Current fouling levels pose a {urgency.lower()} risk to vessel performance. Increased drag from biofouling can lead to:
- 5-15% increase in fuel consumption
- Reduced maneuverability
- Potential corrosion issues
- Risk of invasive species transfer

**Recommended Mitigation Strategies:**
1. **Mechanical Cleaning**: Schedule hull cleaning within 7-14 days using high-pressure water systems
2. **Anti-fouling Coating**: Apply copper-based anti-fouling paint after cleaning
3. **Regular Monitoring**: Implement monthly inspection schedule

**Maintenance Schedule:**
- Immediate Action: Schedule cleaning within 7 days (High Priority)
- Secondary Action: Apply anti-fouling coating within 30 days (Medium Priority)
- Follow-up: Next inspection in 90 days (Low Priority)

**Cost-Benefit Analysis:**
Immediate cleaning cost: {fouling_data.get('Maintenance Cost', '$500-1000')}. Annual fuel savings potential: $2000-5000 with regular maintenance.
"""

async def generate_llm_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate analysis using Hugging Face OpenAI-compatible API"""
    try:
        # Set the environment variable for the API key
        os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
        
        # Initialize the client
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )
        
        # Prepare data points from session analytics
        analytics = data.get("analytics", {})
        fouling_data = {
            "Overall Surface Density": f"{analytics.get('total_coverage', 0):.1f}%",
            "Dominant Species": analytics.get('dominant_species', 'None detected'),
            "Species Count": str(analytics.get('species_count', 0)),
            "Average Confidence": f"{analytics.get('avg_confidence', 0):.2f}",
            "Cleaning Urgency": analytics.get('cleaning_urgency', 'Unknown'),
            "Fuel Cost Impact": f"${analytics.get('fuel_cost_impact', 0):.2f}",
            "Maintenance Cost": f"${analytics.get('maintenance_cost', 0):.2f}"
        }
        
        # Create structured prompt
        structured_prompt = f"""
Generate a professional, multi-section report on marine biofouling.

**Session Information:**
- Session Name: {data.get('session_name', 'Unknown')}
- Date of Analysis: {data.get('created_at', 'Unknown')}

**Input Data:**
- Overall Surface Density: {fouling_data['Overall Surface Density']}
- Dominant Species: {fouling_data['Dominant Species']}
- Species Count: {fouling_data['Species Count']}
- Average Confidence: {fouling_data['Average Confidence']}
- Cleaning Urgency: {fouling_data['Cleaning Urgency']}
- Fuel Cost Impact: {fouling_data['Fuel Cost Impact']}
- Maintenance Cost: {fouling_data['Maintenance Cost']}

**Required Report Sections:**
1. **Executive Summary:** A brief overview of the findings and key concerns.
2. **Detailed Analysis:** Elaborate on the significance of the given data percentages. Discuss the fouling levels and their implications.
3. **Risk Assessment:** Detail the potential risks associated with these levels of fouling, including increased drag, fuel consumption, corrosion, and transfer of invasive species.
4. **Recommended Mitigation Strategies:** Suggest at least two distinct strategies for cleaning and prevention (e.g., mechanical cleaning, anti-fouling coatings). Be specific.
5. **Maintenance Schedule:** Provide a timeline for recommended maintenance actions based on the current fouling levels.
"""
        
        logger.info("Generating LLM report... This may take a moment.")
        
        # Try multiple models for better reliability
        models_to_try = [
            "meta-llama/Llama-3.1-8B-Instruct:nebius",  # Fast and reliable
            "microsoft/DialoGPT-medium:nebius",  # Backup option
            "openai/gpt-oss-120b:nebius"  # Original model as last resort
        ]
        
        raw_report_text = None
        
        for model in models_to_try:
            try:
                logger.info(f"Trying model: {model}")
                
                # Create the API call
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": structured_prompt
                        }
                    ],
                    max_tokens=800,
                    temperature=0.7,
                    timeout=20,  # Shorter timeout for faster failure
                )
                
                # Extract the raw text content from the response
                raw_report_text = completion.choices[0].message.content
                logger.info(f"Successfully generated report with model: {model}")
                break
                
            except Exception as model_error:
                logger.warning(f"Model {model} failed: {model_error}")
                continue
        
        # If all models fail, generate a fallback report
        if raw_report_text is None:
            logger.error("All LLM models failed, generating fallback report")
            raw_report_text = generate_fallback_report(fouling_data, data)
        
        # Format the report with additional context
        formatted_report = format_report_content(raw_report_text, fouling_data, data)
        
        return {
            "analysis": formatted_report,
            "timestamp": datetime.now().isoformat(),
            "model_used": "openai/gpt-oss-120b:nebius",
            "raw_content": raw_report_text
        }
        
    except Exception as e:
        logger.error(f"Failed to generate LLM analysis: {e}")
        # Return fallback analysis
        analytics = data.get("analytics", {})
        return {
            "analysis": f"Error generating analysis: {str(e)}. Fallback analysis: Session {data.get('session_name', 'Unknown')} shows {analytics.get('total_coverage', 0):.1f}% fouling coverage.",
            "timestamp": datetime.now().isoformat(),
            "model_used": "error"
        }
