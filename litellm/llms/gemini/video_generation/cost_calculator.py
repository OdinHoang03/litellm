"""
Gemini Video Generation Cost Calculator for Veo models

This module provides cost calculation functionality for Gemini's Veo video generation models.
All Veo models use a fixed cost of $0.50 per request regardless of video duration.
"""

from typing import Any, Optional

import litellm
from litellm._logging import verbose_logger


def cost_calculator(
    model: str,
    video_response: Optional[Any] = None,
    request_body: Optional[dict] = None,
    video_duration_seconds: Optional[float] = None,
) -> float:
    """
    Gemini Video Generation Cost Calculator for Veo models
    
    Args:
        model: Model name (e.g., "veo-3.0-generate-preview")
        video_response: Video generation response (optional, not used for cost calc)
        request_body: Original request body to extract duration parameters
        video_duration_seconds: Explicit video duration in seconds
    
    Returns:
        float: Cost in USD for the video generation
    """
    # Check if this is a video generation model (Veo models)
    video_model_patterns = ["veo-2", "veo-3", "veo2", "veo3", "video-generation"]
    is_video_model = any(pattern in model.lower() for pattern in video_model_patterns)
    
    if is_video_model:
        # Fixed cost per request for video generation - $0.50 for all Veo models
        total_cost = 0.5
        verbose_logger.info(
            f"🎬💰 VIDEO COST CALCULATED: model={model}, fixed_cost=${total_cost}"
        )
        return total_cost
    
    # For non-video models, try to get standard model info
    try:
        _model_info = litellm.get_model_info(
            model=model,
            custom_llm_provider="gemini",
        )
        # If we can get model info, this isn't a video model, return 0.0
        verbose_logger.warning(
            f"Non-video model {model} passed to video cost calculator. Returning 0.0 cost."
        )
        return 0.0
    except Exception as e:
        # If we can't get model info and it's not a recognized video model, return 0.0
        verbose_logger.warning(
            f"Could not get model info for {model}: {e}. Not a recognized video model, returning 0.0 cost."
        )
        return 0.0

    # This should never be reached, but just in case
    total_cost = 0.0
    
    verbose_logger.debug(
        f"Video generation cost: model={model}, fixed_cost=${total_cost}"
    )
    
    return total_cost


def _extract_video_duration_from_request(request_body: dict) -> Optional[float]:
    """
    Extract video duration from Veo request body
    
    Args:
        request_body: The original request body sent to Veo API
        
    Returns:
        Optional[float]: Video duration in seconds, None if not found
    """
    try:
        if "instances" in request_body:
            instances = request_body["instances"]
            if isinstance(instances, list) and len(instances) > 0:
                instance = instances[0]
                
                # Common parameter names for video duration
                duration_keys = ["duration", "videoLength", "length", "video_duration"]
                
                for key in duration_keys:
                    if key in instance:
                        duration_value = instance[key]
                        if isinstance(duration_value, (int, float)) and duration_value > 0:
                            return float(duration_value)
                
                # Check in video configuration if present
                if "video" in instance:
                    video_config = instance["video"]
                    for key in duration_keys:
                        if key in video_config:
                            duration_value = video_config[key]
                            if isinstance(duration_value, (int, float)) and duration_value > 0:
                                return float(duration_value)
    
    except Exception as e:
        verbose_logger.debug(f"Error extracting video duration from request: {e}")
    
    return None