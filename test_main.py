import torch
from core.forward_analyzer import ForwardAnalyzer
from core.backward_analyzer import BackwardAnalyzer
from models.models import get_model
from utils.visualization import visualize_slice
import pprint

model = get_model("papernetwork")

# STEP 1: Profile the model, set all to zero as in the paper
layer_means = {
    "input": torch.zeros(2),   # <-- REQUIRED NOW
    "fc1": torch.zeros(3),
    "fc2": torch.zeros(2),
    "fc3": torch.zeros(2)
}

# STEP 2: Perform forward analysis
sample = torch.tensor([[1.0, 2.0]])
analyzer = ForwardAnalyzer(model, layer_means)
graph = analyzer.analyze(sample)

# STEP 3: Backward slice
backward_analyzer = BackwardAnalyzer(graph, theta=0.2)
slice_result = backward_analyzer.compute_slice(
    target_neurons=["fc3.neuron_0"],
    input_tensor=sample
)

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(slice_result)

# STEP 4: Visualize slice
visualize_slice(graph, slice_result, input_tensor=sample)