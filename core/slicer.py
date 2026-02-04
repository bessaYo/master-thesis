# core/slicer.py
import torch.nn as nn

from core.tracing.profiler import Profiler
from core.tracing.forward import ForwardAnalyzer
from core.tracing.backward import BackwardAnalyzer
from core.graph import Graph


class Slicer:
    def __init__(self, model: nn.Module, input_sample=None, precomputed_profile=None, debug=False):
        self.model = model
        self.input_sample = input_sample
        self.precomputed_profile = precomputed_profile
        self.graph = None
        self.debug = debug
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
            raise RuntimeError(
                "No profiling samples and no precomputed profile provided."
            )

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
        channel_alpha: float = 0.8,
        block_beta: float = 0.9,
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
            channel_alpha=channel_alpha,
            block_beta=block_beta,
            debug=self.debug,
        )

        backward_analyzer.trace()

        # Neuron count
        total_neurons = sum(c.numel() for c in backward_analyzer.neuron_contributions.values())

        slice_neurons = sum((c != 0).sum().item() for c in backward_analyzer.neuron_contributions.values())

        # Synapse count
        all_synapses = set()
        for layer_name, syns in backward_analyzer.synapse_contributions.items():
            for s in syns:
                # Unique key: (layer, input_idx, output_idx)
                all_synapses.add((layer_name, s["i"], s["j"]))

        # Block info
        skip_blocks = []
        kept_blocks = []
        if backward_analyzer.block_filter and backward_analyzer.blocks:
            skip_blocks = list(backward_analyzer.block_filter.get_skip_blocks())
            kept_blocks = [
                b for b in backward_analyzer.blocks.keys() if b not in skip_blocks
            ]


        self.backward_result = {
            "neuron_contributions": backward_analyzer.neuron_contributions,
            "synapse_contributions": backward_analyzer.synapse_contributions,
            "backward_time_sec": backward_analyzer.backward_time_sec,
            # Slice size metrics
            "total_synapses": self._count_total_synapses(),
            "slice_synapses": len(all_synapses),
            "total_neurons": total_neurons,
            "slice_neurons": slice_neurons,
            # Block info
            "total_blocks": (
                len(backward_analyzer.blocks) if backward_analyzer.blocks else 0
            ),
            "skipped_blocks": len(skip_blocks),
            # Config
            "config": {
                "target_index": target_index,
                "theta": theta,
                "channel_mode": channel_mode,
                "block_mode": block_mode,
                "channel_alpha": channel_alpha,
                "block_beta": block_beta,
            },
        }
        return self.backward_result

    def _count_total_synapses(self):
        """
        Count total *logical* synapses in the model.
        A synapse = connection between two neurons (channels),
        not individual kernel weights.
        """
        total = 0
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                total += module.in_channels * module.out_channels
            elif isinstance(module, nn.Linear):
                total += module.in_features * module.out_features
        return total

    def _build_graph(self):
        self.graph = Graph(self.model)
