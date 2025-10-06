import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as GeminiModelResponseIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class GeminiPassthroughLoggingHandler:
    @staticmethod
    def gemini_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        if "generateContent" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            # Use Gemini config for transformation
            instance_of_gemini_llm = litellm.GoogleAIStudioGeminiConfig()
            litellm_model_response: ModelResponse = instance_of_gemini_llm.transform_response(
                model=model,
                messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
                raw_response=httpx_response,
                model_response=litellm.ModelResponse(),
                logging_obj=logging_obj,
                optional_params={},
                litellm_params={},
                api_key="",
                request_data={},
                encoding=litellm.encoding,
            )
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }
        elif "predictLongRunning" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)
            
            # Create mock ModelResponse for cost tracking
            mock_video_response = GeminiPassthroughLoggingHandler._create_video_generation_mock_response(
                response_body=response_body,
                model=model,
                logging_obj=logging_obj,
            )
            
            # Handle video generation request logging
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_video_generation_logging_payload(
                response_body=response_body,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                request_body=request_body,
                custom_llm_provider="gemini",
            )

            return {
                "result": mock_video_response,  # Return mock response for cost tracking
                "kwargs": kwargs,
            }
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    @staticmethod
    def _handle_logging_gemini_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        model: Optional[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Gemini passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        kwargs: Dict[str, Any] = {}
        model = model or GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)
        complete_streaming_response = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            url_route=url_route,
        )

        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Gemini passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }

        kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="gemini",
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        parsed_chunks = []
        if "generateContent" in url_route or "streamGenerateContent" in url_route:
            gemini_iterator: Any = GeminiModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
                logging_obj=litellm_logging_obj,
            )
            chunk_parsing_logic: Any = gemini_iterator._common_chunk_parsing_logic
            parsed_chunks = [chunk_parsing_logic(chunk) for chunk in all_chunks]
        else:
            return None

        if len(parsed_chunks) == 0:
            return None

        all_openai_chunks = []
        for parsed_chunk in parsed_chunks:
            if parsed_chunk is None:
                continue
            all_openai_chunks.append(parsed_chunk)

        complete_streaming_response = litellm.stream_chunk_builder(chunks=all_openai_chunks)

        return complete_streaming_response

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"

    @staticmethod
    def _create_gemini_response_logging_payload_for_generate_content(
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Gemini passthrough generateContent (streaming and non-streaming)
        """

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider="gemini",
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = custom_llm_provider

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", json.dumps(kwargs, indent=4))

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        logging_obj.model_call_details["response_cost"] = response_cost
        return kwargs

    @staticmethod
    def _create_gemini_video_generation_logging_payload(
        response_body: dict,
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        request_body: dict,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Gemini passthrough video generation (predictLongRunning)
        
        For Veo models, we calculate cost based on video generation parameters
        """
        try:
            # Calculate video generation cost using dedicated calculator
            response_cost = GeminiPassthroughLoggingHandler._calculate_video_generation_cost(
                model=model,
                request_body=request_body,
                response_body={},  # Not used in cost calculation
                start_time=start_time,
                end_time=end_time,
            )

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = custom_llm_provider
            kwargs["call_type"] = "video_generation"

            # Extract prompt for logging
            prompt = ""
            if request_body and "instances" in request_body:
                instances = request_body["instances"]
                if isinstance(instances, list) and len(instances) > 0:
                    prompt = instances[0].get("prompt", "")
            
            kwargs["prompt"] = prompt

            # pretty print standard logging object
            verbose_proxy_logger.debug("Video generation kwargs= %s", json.dumps(kwargs, indent=4, default=str))

            # set litellm_call_id to logging response object
            logging_obj.litellm_call_id = logging_obj.litellm_call_id
            logging_obj.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            logging_obj.model_call_details["response_cost"] = response_cost
            logging_obj.model_call_details["call_type"] = "video_generation"
            
            return kwargs
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error creating Gemini video generation logging payload: %s", e
            )
            return kwargs

    @staticmethod
    def _calculate_video_generation_cost(
        model: str,
        request_body: dict,
        response_body: dict,
        start_time: datetime,
        end_time: datetime,
    ) -> float:
        """
        Calculate cost for Gemini video generation (Veo models)
        
        Uses the dedicated Gemini video generation cost calculator.
        """
        try:
            from litellm.llms.gemini.video_generation.cost_calculator import cost_calculator
            
            verbose_proxy_logger.info(
                f"🎬 Calculating video generation cost for model: {model}"
            )
            
            cost = cost_calculator(
                model=model,
                request_body=request_body,
            )
            
            verbose_proxy_logger.info(
                f"💰 Video generation cost calculated: ${cost} for model {model}"
            )
            
            return cost
            
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error calculating video generation cost for model %s: %s", model, e
            )
            return 0.0

    @staticmethod
    def _create_video_generation_mock_response(
        response_body: dict,
        model: str,
        logging_obj: LiteLLMLoggingObj,
    ) -> ModelResponse:
        """
        Create a mock ModelResponse for video generation to enable cost tracking
        
        Video generation responses don't follow the standard ModelResponse format,
        but we need a ModelResponse object for the cost tracking system to work.
        """
        try:
            # Extract operation name from response
            operation_name = ""
            if "name" in response_body:
                operation_name = response_body["name"]
            elif isinstance(response_body.get("response"), str):
                # Parse JSON string response
                import json
                try:
                    parsed_response = json.loads(response_body["response"])
                    operation_name = parsed_response.get("name", "")
                except json.JSONDecodeError:
                    pass
            
            # Calculate the cost for this video generation request
            calculated_cost = GeminiPassthroughLoggingHandler._calculate_video_generation_cost(
                model=model,
                response_body=response_body,
                request_body={},  # We can pass empty since we use fixed cost
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            
            # Create mock ModelResponse
            mock_response = ModelResponse(
                id=logging_obj.litellm_call_id,
                object="video_generation",
                created=int(datetime.now().timestamp()),
                model=model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"{operation_name}"
                        },
                        "finish_reason": "stop"
                    }
                ],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            )
            
            # Add the calculated cost to hidden params so it gets picked up by the cost calculator
            mock_response._hidden_params = {"response_cost": calculated_cost}
            
            verbose_proxy_logger.info(
                f"🎬💰 VIDEO COST ATTACHED: model={model}, cost=${calculated_cost}, operation={operation_name}"
            )
            
            return mock_response
            
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error creating mock video generation response: %s", e
            )
            # Fallback to basic ModelResponse with fixed cost
            fallback_response = ModelResponse(
                id=logging_obj.litellm_call_id,
                object="video_generation",
                created=int(datetime.now().timestamp()),
                model=model,
                choices=[],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            
            # Still apply the cost even in fallback case
            try:
                fallback_cost = GeminiPassthroughLoggingHandler._calculate_video_generation_cost(
                    model=model,
                    response_body=response_body,
                    request_body={},
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
                fallback_response._hidden_params = {"response_cost": fallback_cost}
                verbose_proxy_logger.info(
                    f"🎬💰 FALLBACK VIDEO COST ATTACHED: model={model}, cost=${fallback_cost}"
                )
            except Exception as cost_error:
                verbose_proxy_logger.warning(
                    f"Failed to calculate fallback cost for {model}: {cost_error}"
                )
                fallback_response._hidden_params = {"response_cost": 0.5}  # Default to $0.50
            
            return fallback_response
