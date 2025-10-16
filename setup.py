#!/usr/bin/env python3
"""
Setup script for Image Classification Project.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Image Classification Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_command} install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    if not run_command(f"{pip_command} install -r requirements.txt", "Installing requirements"):
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "models", "logs", "results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Test installation
    print("\n🧪 Testing installation...")
    test_command = f"{pip_command} install pytest && python -m pytest tests/ -v"
    if run_command(test_command, "Running tests"):
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed, but installation may still work")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("\n2. Run the demo:")
    print("   python demo.py")
    print("\n3. Train a model:")
    print("   python train.py --model-type cnn")
    print("\n4. Launch the web interface:")
    print("   streamlit run web_app/app.py")
    print("\n5. Classify an image:")
    print("   python predict.py <image_path> --model models/cnn_model_final.h5")


if __name__ == "__main__":
    main()
