from setuptools import setup, find_packages

setup(
    name="YourPackageName",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    url="http://yourpackage.homepage/",
    # license="LICENSE.txt",
    description="An awesome package doing awesome things.",
    long_description=open("README.md").read(),
    install_requires=[
        # List all your project's dependencies here.
        # For example:
        # 'numpy>=1.13.3',
    ],
    python_requires=">=3.6",
)
