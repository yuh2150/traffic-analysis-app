from .sparsity import calculate_model_sparsity, print_layer_sparsity
from .weight_analysis import analyze_weight_distribution
from .artifact_manager import ArtifactManager


def __getattr__(name):
    if name == "TrafficTracker":
        from .tracker import TrafficTracker
        return TrafficTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
