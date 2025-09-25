"""
Smart LLM Service - Hybrid approach with instant fallback
Tries external API first, falls back to instant generation if needed
"""

import asyncio
from datetime import datetime
from typing import Dict, Any
import logging
from fast_llm_service import generate_instant_report

logger = logging.getLogger(__name__)

async def generate_smart_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Smart report generation:
    1. Try external LLM API with timeout
    2. Fall back to instant generation if API fails
    """
    try:
        # Try external API with a short timeout
        logger.info("Attempting external LLM API...")
        
        # Import the external LLM service
        from llm_service import generate_llm_analysis
        
        # Set a timeout for the external API call
        try:
            report_content = await asyncio.wait_for(
                generate_llm_analysis(data), 
                timeout=15.0  # 15 second timeout
            )
            logger.info("External LLM API succeeded")
            return report_content
            
        except asyncio.TimeoutError:
            logger.warning("External LLM API timed out, falling back to instant generation")
            return generate_instant_report(data)
            
        except Exception as api_error:
            logger.warning(f"External LLM API failed: {api_error}, falling back to instant generation")
            return generate_instant_report(data)
            
    except Exception as e:
        logger.error(f"Smart LLM service failed: {e}, using instant generation")
        return generate_instant_report(data)
