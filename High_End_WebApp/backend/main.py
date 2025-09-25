"""
Marine Biofouling Detection Backend API
FastAPI-based backend for image processing, ML model inference, and data management
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image
import io
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
import requests
from pydantic import BaseModel
from openai import OpenAI

# Import our custom modules
from preprocessing_service import PreprocessingService
from model_service import ModelService
from database import Database
from utils import generate_session_id, save_image, load_image
from fast_llm_service import generate_instant_report
from image_classification_api import classification_api
from invasive_species_service import invasive_species_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marine Biofouling Detection API",
    description="AI-powered marine biofouling detection and analysis system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
preprocessing_service = PreprocessingService()
model_service = ModelService()
database = Database()

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/api/models", StaticFiles(directory="."), name="models")

# Pydantic models
class SessionCreate(BaseModel):
    session_name: str
    model_name: str = "biofouling-detector-v1"
    confidence_threshold: float = 0.5

class LLMReportRequest(BaseModel):
    session_id: str
    analysis_data: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await database.initialize()
        await model_service.load_model()
        logger.info("Backend services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Marine Biofouling Detection API", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "preprocessing": "ready",
            "model": "ready" if model_service.model_loaded else "loading",
            "database": "ready"
        }
    }

@app.post("/api/sessions")
async def create_session(request: SessionCreate):
    """Create a new analysis session"""
    try:
        session_id = generate_session_id()
        
        session_data = {
            "session_id": session_id,
            "session_name": request.session_name,
            "model_name": request.model_name,
            "confidence_threshold": request.confidence_threshold,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "images": [],
            "results": None
        }
        
        await database.save_session(session_data)
        
        return {
            "session_id": session_id,
            "session_name": request.session_name,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.post("/api/sessions/{session_id}/upload")
async def upload_and_process_images(
    session_id: str,
    files: List[UploadFile] = File(...)
):
    """Upload images and automatically process them through the full pipeline"""
    try:
        session_data = await database.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        uploaded_files = []
        all_results = []
        
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
            
            # Read image data
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Save original image
            original_path = f"uploads/{session_id}_{file.filename}"
            await save_image(image, original_path)
            
            # Auto-preprocess image
            logger.info(f"Preprocessing image: {file.filename}")
            preprocessed_image = await preprocessing_service.preprocess_image(image)
            preprocessed_path = f"processed/{session_id}_{file.filename}"
            await save_image(preprocessed_image, preprocessed_path)
            
            # Run ML analysis
            logger.info(f"Running ML analysis on: {file.filename}")
            analysis_result = await model_service.analyze_image(
                preprocessed_image,
                confidence_threshold=session_data.get("confidence_threshold", 0.5)
            )
            
            # Save segmentation mask
            if analysis_result.get("segmentation_mask") is not None:
                mask_path = f"outputs/{session_id}_mask_{file.filename}"
                await save_image(analysis_result["segmentation_mask"], mask_path)
            
            # Run additional API-based classification
            try:
                api_classification = classification_api.classify_image(original_path, "resnet50")
                analysis_result["api_classification"] = api_classification
                logger.info("API classification completed successfully")
                
                # Check for invasive species if classification was successful
                if api_classification.get('success', False):
                    classification_results = [api_classification]
                    invasive_alerts = invasive_species_service.detect_invasive_species(classification_results)
                    if invasive_alerts:
                        analysis_result["invasive_alerts"] = invasive_alerts
                        logger.warning(f"Invasive species detected in {file.filename}: {len(invasive_alerts)} alerts")
                        
                        # Log critical alerts immediately
                        for alert in invasive_alerts:
                            if alert['risk_level'] == 'critical':
                                logger.critical(f"🚨 CRITICAL INVASIVE SPECIES ALERT: {alert['species_name']} detected with {alert['confidence']:.2f} confidence in {file.filename}")
                            elif alert['risk_level'] == 'high':
                                logger.warning(f"⚠️ HIGH RISK INVASIVE SPECIES: {alert['species_name']} detected with {alert['confidence']:.2f} confidence in {file.filename}")
                
            except Exception as e:
                logger.warning(f"API classification failed: {e}")
                analysis_result["api_classification"] = {"success": False, "error": str(e)}
            
            # Store image metadata
            image_metadata = {
                "filename": file.filename,
                "original_path": original_path,
                "preprocessed_path": preprocessed_path,
                "segmentation_path": f"outputs/{session_id}_mask_{file.filename}",
                "source": "manual",
                "uploaded_at": datetime.now().isoformat(),
                "size": len(image_data),
                "dimensions": image.size
            }
            
            uploaded_files.append(image_metadata)
            all_results.append({
                "image_id": file.filename,
                "detections": analysis_result["detections"],
                "total_coverage": analysis_result["total_coverage"],
                "dominant_species": analysis_result["dominant_species"],
                "processing_time": analysis_result["processing_time"]
            })
        
        # Calculate session-level analytics
        session_analytics = calculate_session_analytics(all_results)
        
        # Update session with results
        await database.update_session_results(session_id, {
            "images": uploaded_files,
            "analysis_results": all_results,
            "analytics": session_analytics,
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        })
        
        return {
            "session_id": session_id,
            "uploaded_files": uploaded_files,
            "analysis_results": all_results,
            "analytics": session_analytics,
            "message": f"Successfully processed {len(uploaded_files)} images"
        }
    except Exception as e:
        logger.error(f"Failed to upload and process images: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload and process images")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and results"""
    try:
        session_data = await database.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_data
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")

@app.get("/api/sessions/{session_id}/area-ratios")
async def get_area_ratios(session_id: str):
    """Get area ratio data for 3D visualization"""
    try:
        session_data = await database.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Extract area ratios from session analytics or generate mock data
        area_ratios = {}
        
        if session_data.get('analytics'):
            analytics = session_data['analytics']
            species_dist = analytics.get('species_distribution', {})
            
            # Convert species distribution to area ratios
            # This is a simplified mapping - in reality, you'd have proper zone mapping
            zone_mapping = {
                'barnacle': 'middle_left_bottom',
                'algae': 'front_right_top', 
                'mussel': 'rear_center',
                'tube_worm': 'top_surface',
                'sponge': 'bottom_hull'
            }
            
            for species, count in species_dist.items():
                zone_name = zone_mapping.get(species.lower(), f"{species.lower()}_zone")
                # Convert count to ratio (0-1)
                ratio = min(count / 100.0, 1.0)  # Normalize to 0-1
                area_ratios[zone_name] = ratio
        
        # If no analytics data, generate mock data based on session results
        if not area_ratios and session_data.get('analysis_results'):
            results = session_data['analysis_results']
            if results:
                # Generate mock area ratios based on analysis confidence
                mock_areas = {
                    'middle_left_bottom': 0.9,
                    'front_right_top': 0.25,
                    'rear_center': 0.6,
                    'top_surface': 0.15,
                    'bottom_hull': 0.8,
                }
                
                # Adjust based on analysis confidence
                avg_confidence = np.mean([r.get('confidence', 0.5) for r in results])
                for zone, base_ratio in mock_areas.items():
                    area_ratios[zone] = base_ratio * avg_confidence
        
        # Default mock data if nothing else is available
        if not area_ratios:
            area_ratios = {
                'middle_left_bottom': 0.9,
                'front_right_top': 0.25,
                'rear_center': 0.6,
                'top_surface': 0.15,
                'bottom_hull': 0.8,
            }
        
        return area_ratios
        
    except Exception as e:
        logger.error(f"Error getting area ratios for session {session_id}: {str(e)}")
        # Return mock data as fallback
        mock_data = {
            'middle_left_bottom': 0.9,
            'front_right_top': 0.25,
            'rear_center': 0.6,
            'top_surface': 0.15,
            'bottom_hull': 0.8,
        }
        return mock_data

@app.get("/api/alerts/invasive-species")
async def get_invasive_species_alerts():
    """Get all invasive species alerts"""
    try:
        # Get all sessions with classification results
        sessions = await database.get_all_sessions()
        all_alerts = []
        
        for session in sessions:
            if session.get('classification_results'):
                alerts = invasive_species_service.detect_invasive_species(
                    session['classification_results']
                )
                # Add session context to alerts
                for alert in alerts:
                    alert['session_id'] = session['session_id']
                    alert['session_name'] = session['session_name']
                    alert['created_at'] = session['created_at']
                
                all_alerts.extend(alerts)
        
        # Sort by detection time (newest first)
        all_alerts.sort(key=lambda x: x['detection_time'], reverse=True)
        
        # Generate summary
        summary = invasive_species_service.get_alert_summary(all_alerts)
        
        return {
            'alerts': all_alerts,
            'summary': summary,
            'total_count': len(all_alerts)
        }
        
    except Exception as e:
        logger.error(f"Error getting invasive species alerts: {str(e)}")
        return {
            'alerts': [],
            'summary': {},
            'total_count': 0,
            'error': 'Failed to retrieve alerts'
        }

@app.get("/api/sessions/{session_id}/alerts")
async def get_session_alerts(session_id: str):
    """Get invasive species alerts for a specific session"""
    try:
        session_data = await database.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        alerts = []
        if session_data.get('classification_results'):
            alerts = invasive_species_service.detect_invasive_species(
                session_data['classification_results']
            )
            # Add session context
            for alert in alerts:
                alert['session_id'] = session_id
                alert['session_name'] = session_data['session_name']
                alert['created_at'] = session_data['created_at']
        
        summary = invasive_species_service.get_alert_summary(alerts)
        
        return {
            'session_id': session_id,
            'alerts': alerts,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session alerts")

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an invasive species alert"""
    try:
        # In a real system, you'd update the alert status in a database
        # For now, we'll return a success response
        return {
            'alert_id': alert_id,
            'status': 'acknowledged',
            'acknowledged_at': datetime.now().isoformat(),
            'message': 'Alert acknowledged successfully'
        }
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@app.get("/api/sessions")
async def get_all_sessions():
    """Get all sessions"""
    try:
        sessions = await database.get_all_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

# Removed old image endpoint - replaced with new structured endpoint below

@app.post("/api/sessions/{session_id}/generate-report")
async def generate_llm_report(session_id: str):
    """Generate LLM-powered PDF report"""
    try:
        session_data = await database.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Prepare data for LLM
        report_data = {
            "session_name": session_data["session_name"],
            "created_at": session_data["created_at"],
            "analytics": session_data.get("analytics", {}),
            "analysis_results": session_data.get("analysis_results", []),
            "images": session_data.get("images", [])
        }
        
        # Generate instant report (fast and reliable)
        report_content = generate_instant_report(report_data)
        
        # Create PDF report
        pdf_path = await create_pdf_report(session_id, report_data, report_content)
        
        return {
            "session_id": session_id,
            "report_path": pdf_path,
            "report_content": report_content,
            "message": "Report generated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

async def old_generate_llm_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate analysis using Hugging Face Inference API"""
    try:
        # Prepare prompt for LLM
        analytics = data.get("analytics", {})
        prompt = f"""
        As a marine biofouling expert, analyze the following data and provide comprehensive recommendations:

        Session: {data.get('session_name', 'Unknown')}
        Date: {data.get('created_at', 'Unknown')}
        
        Analysis Results:
        - Total Fouling Coverage: {analytics.get('total_coverage', 0):.1f}%
        - Species Count: {analytics.get('species_count', 0)}
        - Dominant Species: {analytics.get('dominant_species', 'None')}
        - Average Confidence: {analytics.get('avg_confidence', 0):.2f}
        - Cleaning Urgency: {analytics.get('cleaning_urgency', 'Unknown')}
        - Estimated Fuel Cost Impact: ${analytics.get('fuel_cost_impact', 0):.2f}
        - Estimated Maintenance Cost: ${analytics.get('maintenance_cost', 0):.2f}
        
        Species Distribution:
        {json.dumps(analytics.get('species_distribution', {}), indent=2)}
        
        Please provide:
        1. Executive Summary
        2. Detailed Analysis
        3. Risk Assessment
        4. Maintenance Recommendations
        5. Scheduled Maintenance Timeline
        6. Cost-Benefit Analysis
        7. Action Items
        """
        
        # Use Hugging Face Inference API (you'll need to set your API token)
        headers = {
            "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN', 'your_token_here')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        # For demo purposes, return mock analysis
        return {
            "executive_summary": f"Analysis of {data.get('session_name', 'session')} reveals {analytics.get('total_coverage', 0):.1f}% fouling coverage with {analytics.get('species_count', 0)} different species detected. The dominant species is {analytics.get('dominant_species', 'unknown')}.",
            "detailed_analysis": "Detailed analysis shows moderate to high fouling levels requiring immediate attention. The fouling distribution indicates concentrated growth in specific hull regions.",
            "risk_assessment": f"Current fouling levels pose a {analytics.get('cleaning_urgency', 'medium')} risk to vessel performance, with estimated fuel efficiency loss of {analytics.get('fuel_cost_impact', 0)/10:.1f}%.",
            "maintenance_recommendations": [
                "Schedule immediate hull cleaning within 7-14 days",
                "Focus on high-density fouling areas identified",
                "Implement preventive anti-fouling coating application",
                "Establish regular inspection schedule every 3 months"
            ],
            "scheduled_maintenance": [
                {"task": "Emergency Hull Cleaning", "timeline": "7 days", "priority": "High"},
                {"task": "Anti-fouling Coating", "timeline": "30 days", "priority": "Medium"},
                {"task": "Next Inspection", "timeline": "90 days", "priority": "Low"}
            ],
            "cost_benefit": f"Immediate cleaning cost: ${analytics.get('maintenance_cost', 0):.2f}. Potential fuel savings: ${analytics.get('fuel_cost_impact', 0)*12:.2f} annually.",
            "action_items": [
                "Contact cleaning service provider",
                "Schedule dry dock availability",
                "Order anti-fouling materials",
                "Update maintenance records"
            ]
        }
        
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        # Return fallback analysis
        return {
            "executive_summary": "Analysis completed with basic recommendations.",
            "detailed_analysis": "Detailed analysis based on detected fouling patterns.",
            "risk_assessment": "Moderate risk level identified.",
            "maintenance_recommendations": ["Schedule hull cleaning", "Apply anti-fouling coating"],
            "scheduled_maintenance": [{"task": "Hull Cleaning", "timeline": "14 days", "priority": "High"}],
            "cost_benefit": "Cost-benefit analysis completed.",
            "action_items": ["Schedule maintenance", "Order supplies"]
        }

async def create_pdf_report(session_id: str, data: Dict[str, Any], content: Dict[str, Any]) -> str:
    """Create PDF report with analysis data"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        # Create PDF file
        pdf_path = f"outputs/{session_id}_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1
        )
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("Marine Biofouling Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Session info
        session_info = [
            ["Session Name:", data.get('session_name', 'Unknown')],
            ["Date:", data.get('created_at', 'Unknown')[:10]],
            ["Status:", "Completed"]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(session_table)
        story.append(Spacer(1, 20))
        
        # Analysis Content (from new fast LLM service)
        analysis_content = content.get('analysis', 'No analysis available')
        
        # Extract analytics data for comprehensive report
        analytics = data.get('analytics', {})
        
        # Executive Summary Section
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        exec_summary = f"""
        This comprehensive marine biofouling analysis report provides detailed findings from the inspection of {data.get('session_name', 'Unknown Session')}. 
        The analysis reveals {analytics.get('total_coverage', 0):.1f}% fouling coverage with {analytics.get('species_count', 0)} different species detected. 
        The dominant species is {analytics.get('dominant_species', 'Unknown')}, indicating {analytics.get('cleaning_urgency', 'Medium')} priority cleaning requirements.
        """
        story.append(Paragraph(exec_summary.strip(), styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Detailed Analysis Section
        story.append(Paragraph("Detailed Analysis", styles['Heading2']))
        
        # Create analysis data table
        analysis_data = [
            ["Metric", "Value", "Assessment"],
            ["Overall Surface Density", f"{analytics.get('total_coverage', 0):.1f}%", "High" if analytics.get('total_coverage', 0) > 15 else "Medium" if analytics.get('total_coverage', 0) > 5 else "Low"],
            ["Dominant Species", analytics.get('dominant_species', 'Unknown'), "Identified"],
            ["Species Count", str(analytics.get('species_count', 0)), "Diverse" if analytics.get('species_count', 0) > 3 else "Limited"],
            ["Average Confidence", f"{analytics.get('avg_confidence', 0):.2f}", "High" if analytics.get('avg_confidence', 0) > 0.8 else "Medium"],
            ["Cleaning Urgency", analytics.get('cleaning_urgency', 'Medium'), "Priority Level"],
            ["Fuel Cost Impact", f"${analytics.get('fuel_cost_impact', 0):.2f}", "Annual Impact"],
            ["Maintenance Cost", f"${analytics.get('maintenance_cost', 0):.2f}", "Estimated Cost"]
        ]
        
        analysis_table = Table(analysis_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(analysis_table)
        story.append(Spacer(1, 20))
        
        # Risk Assessment Section
        story.append(Paragraph("Risk Assessment", styles['Heading2']))
        coverage = analytics.get('total_coverage', 0)
        risk_level = "High" if coverage > 15 else "Medium" if coverage > 5 else "Low"
        
        risk_content = f"""
        Current fouling levels pose a {risk_level.lower()} risk to vessel performance and operational efficiency. 
        The {coverage:.1f}% surface coverage indicates significant marine growth that can lead to:
        
        • Increased drag and reduced vessel speed (estimated 5-15% performance loss)
        • Higher fuel consumption and operational costs
        • Potential hull corrosion and structural damage
        • Risk of invasive species transfer to other marine environments
        • Reduced maneuverability in critical situations
        
        Based on the analysis, immediate action is {'required' if coverage > 15 else 'recommended' if coverage > 5 else 'optional'} to maintain optimal vessel performance.
        """
        story.append(Paragraph(risk_content.strip(), styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Recommended Mitigation Strategies
        story.append(Paragraph("Recommended Mitigation Strategies", styles['Heading2']))
        
        strategies = [
            {
                "strategy": "Immediate Hull Cleaning",
                "description": f"Schedule {'emergency' if coverage > 15 else 'routine'} hull cleaning within {'3-7' if coverage > 15 else '7-14' if coverage > 5 else '30'} days",
                "priority": "High" if coverage > 15 else "Medium",
                "cost": f"${analytics.get('maintenance_cost', 500):.0f} - ${analytics.get('maintenance_cost', 500) * 1.5:.0f}"
            },
            {
                "strategy": "Anti-fouling Coating Application",
                "description": "Apply copper-based anti-fouling paint after cleaning to prevent future growth",
                "priority": "Medium",
                "cost": "$800 - $1,200"
            },
            {
                "strategy": "Regular Monitoring Program",
                "description": "Implement monthly inspection schedule with advanced imaging systems",
                "priority": "Low",
                "cost": "$200 - $400/month"
            },
            {
                "strategy": "Preventive Maintenance",
                "description": "Establish quarterly maintenance schedule with professional cleaning services",
                "priority": "Medium",
                "cost": "$1,000 - $2,000/quarter"
            }
        ]
        
        for i, strategy in enumerate(strategies, 1):
            story.append(Paragraph(f"{i}. {strategy['strategy']}", styles['Heading3']))
            story.append(Paragraph(f"Description: {strategy['description']}", styles['Normal']))
            story.append(Paragraph(f"Priority: {strategy['priority']}", styles['Normal']))
            story.append(Paragraph(f"Estimated Cost: {strategy['cost']}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 20))
        
        # Maintenance Schedule
        story.append(Paragraph("Maintenance Schedule", styles['Heading2']))
        
        schedule_data = [["Task", "Timeline", "Priority", "Cost", "Responsible Party"]]
        
        if coverage > 15:
            schedule_data.append(["Emergency Hull Cleaning", "3-7 days", "High", f"${analytics.get('maintenance_cost', 800):.0f}", "Marine Services"])
            schedule_data.append(["Anti-fouling Coating", "14 days", "High", "$1,000", "Coating Specialists"])
        elif coverage > 5:
            schedule_data.append(["Hull Cleaning", "7-14 days", "Medium", f"${analytics.get('maintenance_cost', 600):.0f}", "Marine Services"])
            schedule_data.append(["Anti-fouling Coating", "30 days", "Medium", "$1,000", "Coating Specialists"])
        else:
            schedule_data.append(["Routine Cleaning", "30 days", "Low", f"${analytics.get('maintenance_cost', 400):.0f}", "Marine Services"])
            schedule_data.append(["Coating Assessment", "60 days", "Low", "$200", "Inspector"])
        
        schedule_data.extend([
            ["Next Inspection", "90 days", "Medium", "$300", "Marine Inspector"],
            ["Annual Maintenance", "365 days", "Medium", "$2,500", "Marine Services"],
            ["Performance Review", "180 days", "Low", "$500", "Operations Team"]
        ])
        
        schedule_table = Table(schedule_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1.5*inch])
        schedule_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(schedule_table)
        story.append(Spacer(1, 20))
        
        # Cost-Benefit Analysis
        story.append(Paragraph("Cost-Benefit Analysis", styles['Heading2']))
        
        immediate_cost = analytics.get('maintenance_cost', 500)
        annual_fuel_savings = analytics.get('fuel_cost_impact', 0) * 12
        roi = (annual_fuel_savings / immediate_cost * 100) if immediate_cost > 0 else 0
        break_even = (immediate_cost / annual_fuel_savings * 12) if annual_fuel_savings > 0 else 0
        
        cost_benefit_content = f"""
        Financial Analysis Summary:
        
        • Immediate Cleaning Cost: ${immediate_cost:.0f}
        • Annual Fuel Savings: ${annual_fuel_savings:.0f}
        • Return on Investment: {roi:.1f}%
        • Break-even Period: {break_even:.1f} months
        • 5-Year Net Savings: ${(annual_fuel_savings * 5) - immediate_cost:.0f}
        
        The investment in hull cleaning and maintenance provides significant long-term cost savings through improved fuel efficiency and reduced operational costs.
        """
        story.append(Paragraph(cost_benefit_content.strip(), styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Action Items
        story.append(Paragraph("Action Items", styles['Heading2']))
        
        action_items = [
            "Contact marine cleaning service provider within 24 hours",
            "Schedule dry dock or floating dock availability",
            "Order anti-fouling materials and equipment",
            "Update maintenance records and documentation",
            "Implement monitoring protocol for future inspections",
            "Train crew on fouling identification and reporting",
            "Establish regular maintenance calendar",
            "Review insurance coverage for hull maintenance"
        ]
        
        for i, item in enumerate(action_items, 1):
            story.append(Paragraph(f"{i}. {item}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Add the original LLM analysis if available
        if analysis_content and len(analysis_content) > 100:
            story.append(Paragraph("AI-Generated Detailed Analysis", styles['Heading2']))
            # Clean up the analysis content for PDF
            clean_content = analysis_content.replace('**', '').replace('*', '').strip()
            paragraphs = clean_content.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para and len(para) > 20:
                    story.append(Paragraph(para, styles['Normal']))
                    story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        
        return pdf_path
        
    except ImportError:
        logger.error("ReportLab not installed. Install with: pip install reportlab")
        # Create a simple text report as fallback
        text_path = f"outputs/{session_id}_report.txt"
        with open(text_path, 'w') as f:
            f.write("Marine Biofouling Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Session: {data.get('session_name', 'Unknown')}\n")
            f.write(f"Date: {data.get('created_at', 'Unknown')}\n\n")
            f.write("Executive Summary:\n")
            f.write(content.get('executive_summary', 'No summary available') + "\n\n")
            f.write("Maintenance Recommendations:\n")
            for rec in content.get('maintenance_recommendations', []):
                f.write(f"- {rec}\n")
        
        return text_path
    except Exception as e:
        logger.error(f"Failed to create PDF report: {e}")
        raise

def calculate_session_analytics(analysis_results: List[Dict]) -> Dict[str, Any]:
    """Calculate session-level analytics from detection results"""
    if not analysis_results:
        return {
            "total_coverage": 0,
            "species_count": 0,
            "dominant_species": "None detected",
            "avg_confidence": 0,
            "total_detections": 0,
            "fuel_cost_impact": 0,
            "maintenance_cost": 0,
            "cleaning_urgency": "low"
        }
    
    # Flatten all detections
    all_detections = []
    for result in analysis_results:
        all_detections.extend(result.get("detections", []))
    
    if not all_detections:
        return {
            "total_coverage": 0,
            "species_count": 0,
            "dominant_species": "None detected",
            "avg_confidence": 0,
            "total_detections": 0,
            "fuel_cost_impact": 0,
            "maintenance_cost": 0,
            "cleaning_urgency": "low"
        }
    
    # Calculate coverage statistics
    total_coverage = sum(det["coverage_percentage"] for det in all_detections)
    avg_confidence = sum(det["confidence"] for det in all_detections) / len(all_detections)
    
    # Species distribution
    species_coverage = {}
    for det in all_detections:
        species = det["species"]
        species_coverage[species] = species_coverage.get(species, 0) + det["coverage_percentage"]
    
    dominant_species = max(species_coverage.items(), key=lambda x: x[1])[0] if species_coverage else "None detected"
    
    # Cost calculations (estimated)
    fuel_cost_impact = total_coverage * 2.5  # $2.5 per percentage point
    maintenance_cost = total_coverage * 15   # $15 per percentage point
    
    # Urgency assessment
    if total_coverage > 70:
        cleaning_urgency = "critical"
    elif total_coverage > 40:
        cleaning_urgency = "high"
    elif total_coverage > 20:
        cleaning_urgency = "medium"
    else:
        cleaning_urgency = "low"
    
    return {
        "total_coverage": min(total_coverage, 100),
        "species_count": len(species_coverage),
        "dominant_species": dominant_species,
        "avg_confidence": avg_confidence,
        "total_detections": len(all_detections),
        "fuel_cost_impact": fuel_cost_impact,
        "maintenance_cost": maintenance_cost,
        "cleaning_urgency": cleaning_urgency,
        "species_distribution": species_coverage
    }

@app.get("/api/analytics/charts")
async def get_analytics_charts(time_range: str = Query("30d", description="Time range for analytics")):
    """Get analytics charts data"""
    try:
        # Get all sessions for analytics
        sessions = await database.get_all_sessions()
        
        # Mock analytics data for now
        charts_data = {
            "time_series": {
                "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
                "datasets": [
                    {
                        "label": "Fouling Density",
                        "data": [0.2, 0.4, 0.6, 0.8],
                        "borderColor": "rgb(255, 99, 132)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)"
                    }
                ]
            },
            "species_distribution": {
                "labels": ["Background", "Barnacles", "Mussels", "Seaweed"],
                "datasets": [
                    {
                        "label": "Coverage %",
                        "data": [60, 20, 15, 5],
                        "backgroundColor": [
                            "rgba(54, 162, 235, 0.2)",
                            "rgba(255, 99, 132, 0.2)",
                            "rgba(255, 205, 86, 0.2)",
                            "rgba(75, 192, 192, 0.2)"
                        ]
                    }
                ]
            },
            "cost_analysis": {
                "labels": ["Current", "1 Month", "3 Months", "6 Months"],
                "datasets": [
                    {
                        "label": "Cleaning Cost",
                        "data": [1000, 2000, 4000, 8000],
                        "borderColor": "rgb(255, 99, 132)"
                    }
                ]
            }
        }
        
        return charts_data
    except Exception as e:
        logger.error(f"Failed to get analytics charts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics charts")

@app.post("/api/classify-image")
async def classify_image_api(file: UploadFile = File(...)):
    """Classify image using Hugging Face API"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Classify image using multiple models
        classification_results = classification_api.classify_multiple_models(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "classification_results": classification_results
        }
        
    except Exception as e:
        logger.error(f"Image classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/api/classify-image-single")
async def classify_image_single_api(file: UploadFile = File(...), model: str = "resnet50"):
    """Classify image using a single model"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Classify image using single model
        classification_results = classification_api.classify_image(temp_path, model)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "model_used": model,
            "classification_results": classification_results
        }
        
    except Exception as e:
        logger.error(f"Image classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/api/images/{image_type}/{filename}")
async def get_image(image_type: str, filename: str):
    """Serve images (original, preprocessed, or segmented)"""
    try:
        # Map image types to directories with absolute paths
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        image_dirs = {
            "original": os.path.join(backend_dir, "uploads"),
            "preprocessed": os.path.join(backend_dir, "processed"), 
            "segmented": os.path.join(backend_dir, "outputs"),
            "outputs": os.path.join(backend_dir, "outputs")  # Add outputs for PDFs
        }
        
        if image_type not in image_dirs:
            raise HTTPException(status_code=400, detail="Invalid image type")
        
        image_path = os.path.join(image_dirs[image_type], filename)
        
        logger.info(f"Looking for file: {image_path}")
        logger.info(f"File exists: {os.path.exists(image_path)}")
        
        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        
        # Determine media type based on file extension
        if filename.lower().endswith('.pdf'):
            media_type = "application/pdf"
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            media_type = "image/jpeg"
        else:
            media_type = "application/octet-stream"
        
        # Use FileResponse to serve the file
        return FileResponse(image_path, media_type=media_type)
        
    except Exception as e:
        logger.error(f"Failed to serve file {image_type}/{filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve file")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)