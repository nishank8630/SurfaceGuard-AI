from setuptools import setup, find_packages

setup(
    name="surfaceguard",
    version="1.0.0",
    description="Deep Learning-Based Industrial Defect Detection and Anomaly Localization",
    author="SurfaceGuard AI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "pydantic>=2.0.0",
        "tqdm>=4.66.0",
    ],
)
