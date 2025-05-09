import kagglehub

# Download latest version
path = kagglehub.model_download("google/mobilenet-v3/tfLite/large-100-224-feature-vector")

print("Path to model files:", path)