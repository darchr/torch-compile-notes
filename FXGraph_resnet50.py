# Note: output redirected to output1.txt
import torch

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

from torchvision.models import resnet50, ResNet50_Weights
def init_model():
    return resnet50().to(torch.float32).cuda()

def evaluate(mod, inp):
    return mod(inp)

import torch._dynamo

evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")

inp = generate_data(16)[0]


from typing import List
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Reset since we are using a different backend.
torch._dynamo.reset()

opt_model = torch.compile(init_model(), backend=custom_backend)
opt_model(generate_data(16)[0])