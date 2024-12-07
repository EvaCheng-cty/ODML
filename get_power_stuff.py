import torch
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import os
import re

def load_model_from_txt(model_txt_path, model_weights_path):
    with open(model_txt_path, 'r') as f:
        model_code = f.read()
    
    # Include necessary imports in the exec environment
    exec_globals = {"nn": torch.nn, "torch": torch}
    exec(model_code, exec_globals)  # Dynamically evaluate the model class
    model_class = [v for v in exec_globals.values() if isinstance(v, type) and issubclass(v, torch.nn.Module)][0]
    
    # Load the checkpoint
    checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Determine model type and layer keys
    if "mlp.0.weight" in checkpoint:
        # MLP Model
        first_layer_weight_shape = checkpoint["mlp.0.weight"].shape
        input_dim = first_layer_weight_shape[1]  # Number of input features
        hidden_dim = first_layer_weight_shape[0]  # Number of hidden units
        model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=26)
    elif "layers.5.weight" in checkpoint:
        # CNN Model
        first_layer_weight_shape = checkpoint["layers.0.weight"].shape
        hidden_dim = 256  # Extract hidden_dim from the fully connected layer
        model = model_class(hidden_dim=hidden_dim, num_classes=26)
    else:
        raise KeyError("Unknown model architecture: Could not find matching keys in checkpoint.")

    # Load model state
    model.load_state_dict(checkpoint)
    
    return model


def calculate_energy_consumption(model, input_size, device, model_type):
    if model_type == "MLP":
        # Input for MLP: (batch_size, input_dim)
        input_tensor = torch.randn(input_size[0], input_size[1]).to(device).float()
    elif model_type == "CNN":
        # Input for CNN: (batch_size, channels, height, width)
        input_tensor = torch.randn((1,2,5,5)).to(device).float()
    else:
        raise ValueError("Invalid model type. Must be 'MLP' or 'CNN'.")

    # Track energy consumption
    tracker = EmissionsTracker()
    tracker.start()

    # Run a single inference
    model(input_tensor)

    tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed  # in kWh
    return energy_kwh


def calculate_inferences_under_budget(energy_kwh_per_inference, budget_wh=10):
    budget_kwh = budget_wh / 1000
    return budget_kwh / energy_kwh_per_inference if energy_kwh_per_inference > 0 else 0

def plot_accuracy_vs_inferences(models, budget_wh=10, save_path="accuracy_vs_inferences.png"):
    mlp_inferences = []
    mlp_accuracies = []
    cnn_inferences = []
    cnn_accuracies = []
    mlp_names = []
    cnn_names = []
    
    # Process each model
    for model_name, model_txt_path, model_weights_path, accuracy in models:
        print(f"Processing {model_name}...")
        device = torch.device('cpu')  # Assuming all models run on CPU
        model = load_model_from_txt(model_txt_path, model_weights_path).to(device)

        # Determine model type and input shape
        model_type = "CNN" if re.search(r'\d{4}\.pth$', model_weights_path) else "MLP"
        if model_type == "MLP":
            input_size = (1, 42)  # Batch size = 1, input_dim = 42
        elif model_type == "CNN":
            input_size = (1, 2, 21, 21)  # Batch size = 1, channels = 2, height = 21, width = 21

        # Calculate energy consumption per inference
        energy_kwh = calculate_energy_consumption(model, input_size=input_size, device=device, model_type=model_type)
        energy_kwh_per_inference = energy_kwh
        
        # Calculate number of inferences under the 10 Wh budget
        num_inferences = calculate_inferences_under_budget(energy_kwh_per_inference, budget_wh)
        
        # Separate data based on model type
        if model_type == "MLP":
            mlp_inferences.append(num_inferences)
            mlp_accuracies.append(accuracy)
            mlp_names.append(model_name)
        elif model_type == "CNN":
            cnn_inferences.append(num_inferences)
            cnn_accuracies.append(accuracy)
            cnn_names.append(model_name)

        print(f"{model_name}: Accuracy={accuracy}, Inferences={num_inferences}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Helvetica"

    # MLP line and scatter points
    plt.plot(
        mlp_inferences, mlp_accuracies, 
        marker="o", linestyle="-", color="purple", label="MLP", linewidth=3
    )
    for i, name in enumerate(mlp_names):
        plt.annotate(
            name, 
            (mlp_inferences[i], mlp_accuracies[i]), 
            textcoords="offset points", xytext=(0, 10), 
            ha="center", va="bottom", fontsize=8, rotation=45
        )

    # CNN line and scatter points
    plt.plot(
        cnn_inferences, cnn_accuracies, 
        marker="o", linestyle="-", color="yellow", label="CNN", linewidth=3
    )
    for i, name in enumerate(cnn_names):
        plt.annotate(
            name, 
            (cnn_inferences[i], cnn_accuracies[i]), 
            textcoords="offset points", xytext=(0, 10), 
            ha="center", va="bottom", fontsize=8, rotation=45
        )

    # Formatting
    plt.title("Model Accuracy vs. Number of Inferences under 10 Wh")
    plt.xlabel("Number of Inferences under 10 Wh")
    plt.ylabel("Model Accuracy (%)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="-", alpha=0.3)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(save_path)  # Save the plot as an image
    plt.show()


## Define models with their paths and accuracies
models_dir = "./models/"
models = [
    ("MLP (depth=3)", f"{models_dir}nn_new_dataset_85.txt", f"{models_dir}nn_new_dataset_85.pth", 0.8496),
    ("MLP (depth=4)", f"{models_dir}nn_new_dataset_88.txt", f"{models_dir}nn_new_dataset_88.pth", 0.8797),
    ("MLP (depth=5)", f"{models_dir}nn_new_dataset_879.txt", f"{models_dir}nn_new_dataset_879.pth", 0.8797),
    ("CNN (depth=2, mlpdepth=2)", f"{models_dir}cnn_new_dataset_7067.txt", f"{models_dir}cnn_new_dataset_7067.pth", 0.7076),
    ("CNN (depth=1, mlpdepth=4)", f"{models_dir}cnn_new_dataset_7293.txt", f"{models_dir}cnn_new_dataset_7293.pth", 0.7293),
    ("CNN (depth=2, mlpdepth=4)", f"{models_dir}cnn_new_dataset_7669.txt", f"{models_dir}cnn_new_dataset_7669.pth", 0.7669),
]
# Run the plot function
plot_accuracy_vs_inferences(models)
