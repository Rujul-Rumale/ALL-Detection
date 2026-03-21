import pytest
from src.config import INPUT_RES, CLASS_NAMES

def test_config_constants():
    """Verify that canonical constants are set correctly."""
    assert INPUT_RES == 320
    assert len(CLASS_NAMES) == 2
    assert CLASS_NAMES[0] == "ALL"
    assert CLASS_NAMES[1] == "HEM"

def test_imports():
    """Verify that the components of the canonical pipeline can be imported."""
    try:
        from src.detection.demo_pipeline import DemoPipeline
        from src.utils.preprocessing import resize_with_padding
        from src.ui.classification_demo import ClassificationDemoApp
        from training_scripts.train import get_model, MildFocalLoss
    except ImportError as e:
        pytest.fail(f"Could not import canonical modules: {e}")
