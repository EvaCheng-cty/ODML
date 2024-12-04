import torch.nn as nn
import torch
num_classes = 26  # Adjust based on your dataset
batch_size = 128
learning_rate = 0.0001
weight_decay = 0.0001
epochs = 100
hidden_dim = 1024
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        logits = self.mlp(x)[0]
        probs = torch.softmax(logits, dim=0)
        max_prob, max_index = torch.max(probs, dim=0)
        return max_index, max_prob
    
import coremltools as ct

model = MLPModel(input_dim=42, hidden_dim=hidden_dim, num_classes=num_classes)
model.load_state_dict(torch.load("./nn_new_dataset_879.pth"))

model.eval()

# example_input = torch.rand(1, 42) 
example_input = torch.rand(42) 
traced_model = torch.jit.trace(model, example_input)

# convert to coreml model
model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape, name="input")],
    outputs=[ct.TensorType(name="classLabel"), ct.TensorType(name="classProbability")],
)

model_from_trace.save("./nn_new_dataset_879.mlpackage")
