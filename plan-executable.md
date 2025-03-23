# SpotID Executable Plan

## Required Context Files

To fully understand the scope and requirements of this project, humans/aIs should read:

1. **Core Functionality:**
   - `interface/app.py` - The Flask server that powers the web interface
   - `interface/templates/index.html` - The frontend UI
   - `leopard_id/inference_embeddings.py` - The inference pipeline
   - `leopard_id/config_inference.json` - Model and path configurations

2. **Current Setup Process:**
   - `README.md` - Contains current setup instructions for technical users
   - `requirements.txt` - Lists all Python dependencies

3. **Executable Development:**
   - `launcher.py` - The entry point that will be compiled into the executable
   - `spotid.spec` - PyInstaller configuration for building the executable

[previous content continues below...]

## Introduction

The SpotID project provides a powerful deep learning model for leopard individual identification along with a user-friendly web-based interface. Currently, to use the system, wildlife researchers must:

1. Have Python installed on their computer
2. Create a virtual environment
3. Install various dependencies
4. Run specific commands in the terminal
5. Navigate to a URL in their browser

While these steps are straightforward for technical users, they present a significant barrier for wildlife researchers who may not have technical expertise.

This plan outlines the process of packaging the entire SpotID system into a single executable file (.exe) that will:

1. Contain all required components:
   - Python runtime
   - Required libraries
   - Pre-trained model
   - Web interface
   - Example dataset

2. Require NO technical setup:
   - No Python installation needed
   - No virtual environment creation
   - No package installation
   - No terminal commands

3. Have a single-click execution:
   - Double-click the SpotID icon
   - Web interface automatically opens in the default browser
   - All components run in the background

The end result will be an executable that maintains the exact same functionality as the current system but can be used by anyone, regardless of their technical expertise.

## Required Steps

### 1. Prepare launcher script
- [x] Create launcher.py
- [x] Add proper imports and system path configuration
- [ ] Test launcher script independently

### 2. PyInstaller Configuration
- [x] Create PyInstaller spec file (spotid.spec)
- [ ] Test configuration with a simple build
- [ ] Adjust file paths in spec

### 3. Modify config files
- [x] Update config_inference.json with relative paths:
  ```json
  {
    "model_path": "./weights/best-model-cosface.pth",
    "unprocessed_image_folder": "./data/minimum_train_data_cropped",
    "crop_output_folder": "./data/crop_output",
    "bg_removed_output_folder": "./data/background_removed_output",
    "base_binary_output_folder": "./data/edge_detected_output"
  }
  ```

### 4. Bundle Required Files
- [x] Verify all required files are present:
  - [x] Model weight file (leopard_id/weights/best-model-cosface.pth)
  - [x] Minimum dataset (leopard_id/data/minimum_train_data_cropped)
  - [x] Interface templates and static files
  - [x] Python depedencies from requirements.txt

### 5. Build Process
- [ ] Test build in development environment
- [ ] Fix any missing dependencies or import issues
- [ ] Build final executable

### 6. Testing
- [ ] Test executable on a fresh machine
- [ ] Verify all paths work correctly
- [ ] Test preprocessing pipeline
- [ ] Test model inference
- [ ] Test interface functions

## Expected Output Structure
```
SpotID/
├── spotid.exe
├── weights/
│   └── best-model-cosface.pth
├── data/
│   └── minimum_train_data_cropped/
├── config_inference.json
└── _internal/
    ├── leopard_id python modules
    ├── dependencies
    └── interface assets
```

## Current Progress

✅ Completed:
- Installed PyInstaller
- Created launcher.py with proper system path configuration
- Created PyInstaller spec file

⏳ In Progress:
- Debugging executable path resolution issues

🔲 Next Steps:
1. Fix executable path resolution issues (detailed below)

## Current Status and Issues

### What Works
- Development environment:
  - Running `python launcher.py` works correctly
  - Landing page loads properly
  - Inference embeddings runs successfully when paths are provided
  - Comparison page displays correctly with leopard embeddings

### What Doesn't Work
- Executable version (`./dist/spotid/spotid`):
  - Landing page loads properly
  - `/run_model_from_scratch` endpoint is called (verified through logging)
  - Process gets stuck after this point with no error message
  - No logs from inference_embeddings.py appear, suggesting it's not being executed

### Key Findings
1. The executable correctly bundles all files (interface, model, preprocessing scripts)
2. The web interface part of the application works in the executable
3. The issue appears to be in the subprocess execution of inference_embeddings.py within the bundled environment

### Required Fixes
1. Review subprocess execution in app.py:
   ```python
   if getattr(sys, 'frozen', False):
       # Running in bundle - current approach not working
       script_path = os.path.join(sys._MEIPASS, "leopard_id", "inference_embeddings.py")
   ```
   
2. Potential solutions to investigate:
   - Import and call run_inference directly instead of using subprocess
   - Use PyInstaller hooks to properly include inference_embeddings.py
   - Modify how paths are resolved in the bundled environment
   - Add extensive logging throughout the execution path

3. Testing requirements:
   - Add logging at critical points in both app.py and inference_embeddings.py
   - Test path resolution in bundled environment
   - Verify module imports work in bundled context
