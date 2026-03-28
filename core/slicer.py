from core.tracing.profiler import Profiler
from core.tracing.forward import ForwardAnalyzer
from core.tracing.backward import BackwardAnalyzer
from core.graph import Graph


class Slicer:
    """Main entry point: profile() -> forward() -> backward()"""

    def __init__(self, model, input_sample=None, precomputed_profile=None):
        self.model = model
        self.input_sample = input_sample
        self.precomputed_profile = precomputed_profile
        self.graph = None
        # Result storage
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

        sample = input_sample
        if sample is None:
            sample = self.input_sample
        if sample is None:
            raise RuntimeError("No input sample provided.")

        forward_analyzer = ForwardAnalyzer(self.model, self.profiler_result)
        self.forward_result = forward_analyzer.execute(sample)
        return self.forward_result

    # Phase 3: Backward analysis phase
    def backward(self, target_index=0, theta=0.2, channel_alpha=None, block_beta=None, vectorized=True):
        if self.forward_result is None:
            raise RuntimeError("Run forward() first.")
        if self.graph is None:
            raise RuntimeError("Run profile() first.")

        backward_analyzer = BackwardAnalyzer(
            graph=self.graph,
            forward_result=self.forward_result,
            target_index=target_index,
            theta=theta,
            channel_alpha=channel_alpha,
            block_beta=block_beta,
            vectorized=vectorized,
        )

        backward_analyzer.trace()

        self.backward_result = self._build_summary(
            backward_analyzer,
            target_index,
            theta,
            channel_alpha,
            block_beta,
        )
        return self.backward_result

    def _build_summary(self, analyzer, target_index, theta, channel_alpha, block_beta):
        """Build summary dict from backward analyzer results"""
        # Use neuron_deltas for total count (includes skipped blocks)
        total_neurons = 0
        for d in analyzer.neuron_deltas.values():
            total_neurons += d.numel()
        slice_neurons = 0
        for c in analyzer.neuron_contributions.values():
            slice_neurons += (c != 0).sum().item()

        skip_blocks = []
        if analyzer.block_slicer and analyzer.blocks:
            skip_blocks = list(analyzer.block_slicer.get_skip_blocks())

        return {
            "neuron_contributions": analyzer.neuron_contributions,
            "synapse_contributions": analyzer.synapse_contributions,
            "backward_time": analyzer.backward_time,
            "total_neurons": total_neurons,
            "slice_neurons": slice_neurons,
            "total_blocks": len(analyzer.blocks) if analyzer.blocks else 0,
            "skipped_blocks": len(skip_blocks),
            "skipped_block_names": set(skip_blocks),
            "config": {
                "target_index": target_index,
                "theta": theta,
                "channel_alpha": channel_alpha,
                "block_beta": block_beta,
            },
        }

    def _build_graph(self):
        self.graph = Graph(self.model)
