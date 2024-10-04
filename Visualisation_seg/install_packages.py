import subprocess

packages = [
    "pip",
    "open3d",
    "wandb",
    "PyYAML",
    "multimethod",
    "termcolor",
    "trimesh",
    "easydict",
    "torch",
    "matplotlib",
    "scikit-learn",
    "numpy==1.24"

]

for package in packages:
    subprocess.check_call(["pip", "install", package])
