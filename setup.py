from setuptools import setup, find_packages
import os
import subprocess

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Function to check and install missing requirements
def install_requirements():
    for requirement in requirements:
        try:
            __import__(requirement.split('==')[0])
        except ImportError:
            subprocess.check_call(["pip", "install", requirement])

# Install missing requirements
install_requirements()

# Create an executable script to start the Flask application
entry_point = "app:app"

setup(
    name="flask_workspace",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'start-flask-app = app:app.run',
        ],
    },
    description="A Flask application with required dependencies",
    author="Debargha Ganguly",
    author_email="debargha.ganguly@gmail.com",
    url="http://custom-chatbot.com",
)