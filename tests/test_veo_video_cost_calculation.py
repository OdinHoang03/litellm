"""
Test cases for Veo video generation cost calculation
"""
import pytest
import os
import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.gemini.video_generation.cost_calculator import cost_calculator


@pytest.fixture(autouse=True)
def setup_model_cost():
    """Setup local model cost map for testing"""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")


class TestVeoVideoCostCalculation:
    """Test suite for Veo video generation cost calculation"""

    def test_veo_3_generate_preview_cost(self):
        """Test cost calculation for veo-3.0-generate-preview"""
        model = "veo-3.0-generate-preview"
        
        # Test with explicit duration (should still return fixed cost)
        cost = cost_calculator(model=model, video_duration_seconds=8.0)
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_veo_3_fast_generate_preview_cost(self):
        """Test cost calculation for veo-3.0-fast-generate-preview"""
        model = "veo-3.0-fast-generate-preview"
        
        # Test with explicit duration (should still return fixed cost)
        cost = cost_calculator(model=model, video_duration_seconds=5.0)
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_veo_cost_with_request_body(self):
        """Test cost calculation with duration in request body"""
        model = "veo-3.0-generate-preview"
        request_body = {
            "instances": [{
                "prompt": "A cat playing with a ball of yarn",
                "duration": 10.0
            }]
        }
        
        cost = cost_calculator(model=model, request_body=request_body)
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_veo_default_duration_behavior(self):
        """Test default duration behavior for different Veo models"""
        # Standard model should return fixed cost
        standard_model = "veo-3.0-generate-preview"
        cost = cost_calculator(model=standard_model)
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"
        
        # Fast model should also return fixed cost
        fast_model = "veo-3.0-fast-generate-preview"
        cost = cost_calculator(model=fast_model)
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_veo_request_body_duration_extraction(self):
        """Test that cost is fixed regardless of duration parameters"""
        model = "veo-3.0-generate-preview"
        
        # Test different duration parameter names (should all return same fixed cost)
        test_cases = [
            {"duration": 12.0},
            {"videoLength": 12.0},
            {"length": 12.0},
            {"video_duration": 12.0},
        ]
        
        for params in test_cases:
            request_body = {"instances": [{"prompt": "test", **params}]}
            cost = cost_calculator(model=model, request_body=request_body)
            expected_cost = 0.5  # Fixed cost per request
            assert cost == expected_cost, f"Failed for params {params}: Expected {expected_cost}, got {cost}"

    def test_litellm_completion_cost_video_generation(self):
        """Test video generation cost using litellm.completion_cost"""
        cost = completion_cost(
            model="gemini/veo-3.0-generate-preview",
            call_type="video_generation",
            optional_params={"video_duration_seconds": 8.0}
        )
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_vertex_ai_veo_model_cost(self):
        """Test cost calculation for Vertex AI Veo models"""
        cost = completion_cost(
            model="vertex_ai/veo-3.0-generate-preview",
            call_type="video_generation",
            optional_params={"video_duration_seconds": 6.0}
        )
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_invalid_model_cost_calculation(self):
        """Test cost calculation for models without video generation pricing"""
        # This should return 0.0 for models without video generation pricing
        cost = cost_calculator(model="non-existent-model")
        assert cost == 0.0, f"Expected 0.0 for invalid model, got {cost}"

    def test_zero_duration_cost(self):
        """Test cost calculation with zero duration"""
        model = "veo-3.0-generate-preview"
        # Zero duration should still return fixed cost
        cost = cost_calculator(model=model, video_duration_seconds=0.0)
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost} for zero duration, got {cost}"

    def test_negative_duration_handling(self):
        """Test handling of negative duration values"""
        model = "veo-3.0-generate-preview"
        # Should still return fixed cost for negative values
        cost = cost_calculator(model=model, video_duration_seconds=-5.0)
        expected_cost = 0.5  # Fixed cost per request
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"


if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])