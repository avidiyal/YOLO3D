import onnx

# Load and check the model (silent when successful)
onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)

# Print confirmation
print("Model validation successful!")

# Print basic model info
print(f"IR Version: {onnx_model.ir_version}")
print(f"Producer name: {onnx_model.producer_name}")
print(f"Model version: {onnx_model.model_version}")

# Print input/output information
print("\nModel inputs:")
for input in onnx_model.graph.input:
    print(f"- {input.name}: {input.type.tensor_type.shape}")
    
print("\nModel outputs:")
for output in onnx_model.graph.output:
    print(f"- {output.name}")

print(f"\nNumber of nodes: {len(onnx_model.graph.node)}")