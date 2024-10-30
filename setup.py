from setuptools import setup, find_packages

setup(
    author="Bowen Jin",
    author_email="bowenjin@stu.njmu.edu.cn",
    version="0.0.1",
    name="SSD-py",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "spams",
    ],
    python_requires=">=3",
    description="Python interface for Supervised Sparse Decomposition (SSD)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)