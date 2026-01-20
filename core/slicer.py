# core/slicer.py
import torch.nn as nn

from core.tracing.profiler import Profiler
from core.tracing.forward import ForwardAnalyzer
from core.tracing.backward import BackwardAnalyzer
from core.graph import Graph


class Slicer:
    def __init__(self, model: nn.Module, input_sample=None, precomputed_profile=None):
        self.model = model
        self.input_sample = input_sample
        self.precomputed_profile = precomputed_profile
        self.graph = None

        # Results storage
        self.profiler_result = None
        self.forward_result = None
        self.backward_result = None

    # Phase 1: Profiling phase (average activations)
    def profile(self, profiling_samples=None):
        if self.precomputed_profile is not None:
            self.profiler_result = self.precomputed_profile
            self._build_graph()
            return self.profiler_result

        if profiling_samples is None:
            raise RuntimeError("No profiling samples and no precomputed profile provided.")

        profiler = Profiler(self.model)
        self.profiler_result = profiler.execute(profiling_samples)
        self._build_graph()

        return self.profiler_result

    # Phase 2: Forward analysis phase (deltas from mean activations)
    def forward(self, input_sample=None):
        if self.profiler_result is None:
            raise RuntimeError("Run profile() first.")
        
        sample = input_sample if input_sample is not None else self.input_sample
        if sample is None:
            raise RuntimeError("No input sample provided.")

        forward_analyzer = ForwardAnalyzer(self.model, self.profiler_result)
        self.forward_result = forward_analyzer.execute(sample)
        return self.forward_result

    # Phase 3: Backward analysis phase
    def backward(
        self,
        target_index: int = 0,
        theta: float = 0.3,
        channel_mode: bool = False,
        block_mode: bool = False,
        filter_percent: float = 0.2,
        block_threshold: float = 0.5,
        debug: bool = True,
    ):
        if self.forward_result is None:
            raise RuntimeError("Run forward() first.")
        if self.graph is None:
            raise RuntimeError("Run profile() first.")

        backward_analyzer = BackwardAnalyzer(
            graph=self.graph,
            forward_result=self.forward_result,
            target_index=target_index,
            theta=theta,
            channel_mode=channel_mode,
            block_mode=block_mode,
            filter_percent=filter_percent,
            block_threshold=block_threshold,
            debug=debug,
        )

        backward_analyzer.trace()
        self.backward_result = {
            "neuron_contributions": backward_analyzer.neuron_contributions,
            "synapse_contributions": backward_analyzer.synapse_contributions,
            "backward_time_sec": backward_analyzer.backward_time_sec,
        }

        return self.backward_result

    def _build_graph(self):
        self.graph = Graph(self.model)