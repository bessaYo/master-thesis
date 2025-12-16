# core/slicer.py
import torch.nn as nn

from core.analysis.profiler import Profiler
from core.analysis.forward import ForwardAnalyzer
from core.analysis.backward import BackwardAnalyzer
from core.graph import Graph


class Slicer:
    def __init__(self, model: nn.Module, profiling_samples=None, input_sample=None):
        self.model = model
        self.profiling_samples = profiling_samples
        self.input_sample = input_sample
        self.graph = None

        # Results storage
        self.profiler_result = None
        self.forward_result = None
        self.backward_result = None

    # Phase 1: Profiling phase (average activations)
    def profile(self):
        if self.profiling_samples is None:
            raise RuntimeError(
                "Profiling samples are None. Please provide samples for profiling."
            )

        profiler = Profiler(self.model)
        self.profiler_result = profiler.execute(self.profiling_samples)

        # Build graph after profiling
        self._build_graph()

        return self.profiler_result

    # Phase 2: Forward analysis phase (deltas from mean activations)
    def forward(self):
        if self.profiler_result is None:
            raise RuntimeError(
                "Profiler result is None. Please run profiling phase first."
            )
        if self.input_sample is None:
            raise RuntimeError(
                "Input sample is None. Please provide an input sample for forward analysis."
            )

        forward_analyzer = ForwardAnalyzer(self.model, self.profiler_result)
        self.forward_result = forward_analyzer.execute(self.input_sample)
        return self.forward_result

    # Phase 3: Backward analysis phase (to be implemented)
    def backward(self, target_index=0, theta=0.3, channel_mode=False):
        if self.forward_result is None:
            raise RuntimeError(
                "Forward result is None. Run forward() before backward()."
            )
        if self.graph is None:
            raise RuntimeError("Graph has not been built. Run profile() first.")

        backward_analyzer = BackwardAnalyzer(
            graph=self.graph,
            forward_result=self.forward_result,
            target_index=target_index,
            theta=theta,
            channel_mode=channel_mode,
            debug=True,
        )

        backward_analyzer.trace()
        self.backward_result = {
            "neuron_contributions": backward_analyzer.neuron_contributions,
            "synapse_contributions": backward_analyzer.synapse_contributions,
        }

        return self.backward_result

    # Build graph model
    def _build_graph(self):
        self.graph = Graph(self.model)

    # Execute all phases
    def execute(self, target_index=0, theta=0.3, channel_mode=False):
        if self.profiling_samples is None or self.input_sample is None:
            raise RuntimeError("Provide profiling samples AND input sample.")

        self.profile()
        self.forward()
        self.backward(target_index, theta, channel_mode)

        return {
            "slice": self.backward_result,
        }
