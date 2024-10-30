from setuptools import setup, find_packages

setup(
    name="SSD-python",
    version="0.0.1",
    python_requires='>=3.9.0',
    author="Bowen Jin",
    author_email="bowenjin@stu.njmu.edu.cn",
    description="Python interface for Supervised Sparse Decomposition (SSD)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.3",
        "scipy>=1.5",
        "torch",
        "spams",
    ],
)
