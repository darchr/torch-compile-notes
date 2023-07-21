

# Note: output redirected to output.txt
import torch

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

# N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()

def evaluate(mod, inp):
    return mod(inp)

# model = init_model()

# Reset since we are using a different mode.
import torch._dynamo
# torch._dynamo.reset()

evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")

inp = generate_data(16)[0]
# print("eager:", timed(lambda: evaluate(model, inp))[1])
# print("compile:", timed(lambda: evaluate_opt(model, inp))[1])

from typing import List
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Reset since we are using a different backend.
torch._dynamo.reset()

opt_model = torch.compile(init_model(), backend=custom_backend)
opt_model(generate_data(16)[0])