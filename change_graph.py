import torch
import torch.fx as fx
import torch.nn.functional as F


# Actual Model
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x.size()
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x)
        print(x.size())
        return x

m = Module()
print("original version")
# symbolically trace it
gm = torch.fx.symbolic_trace(m)
# call and print the graph
gm.graph.print_tabular()

def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)

    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        #TODO: size of the tensor
        #TODO: differentiate btw layer and data
        #TODO (if possible): add a new list to the tabular to display the size
        if node.op == 'call_method':
            # The target attribute is the function
            # that call_function calls.
            if node.target == 'size':
                node.target = 'size = 16'

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)

print("manipulated version")
# instantiate it
changed = transform(Module())
# symbolically trace it
gm = torch.fx.symbolic_trace(changed)
# call and print the graph
gm.graph.print_tabular()
