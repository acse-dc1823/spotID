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
- [x] Test launcher script independently

### 2. PyInstaller Configuration
- [x] Create PyInstaller spec file (spotid.spec)
- [x] Test configuration with a simple build
- [x] Adjust file paths in spec

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
- [x] Test build in development environment
- [x] Fix any missing dependencies or import issues
- [x] Build final executable

### 6. Testing
- [x] Test executable on a fresh machine
- [x] Verify all paths work correctly
- [x] Test preprocessing pipeline
- [x] Test model inference
- [x] Test interface functions

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
- Successfully built working executable
- Fixed path resolution issues
- Fixed SSL module integration
- Completed end-to-end testing

⏳ In Progress:
- Creating distribution package

🔲 Next Steps:
1. Package executable for distribution (see Distribution Plan below)

## Current Status

### What Works
- Development environment:
  - Running `python launcher.py` works correctly
  - Landing page loads properly
  - All preprocessing steps (background removal, edge detection) work successfully
  - Full inference pipeline execution including model loading
- Executable version (`./dist/spotid/spotid`):
  - Successfully builds and runs
  - All functionality working end-to-end
  - No SSL or path resolution issues
  - Model weights and dependencies properly bundled

### TO DOs:

- Avoid doubling of images. It seems to compare images across different directories, not only the directory provided. It uses the image in the dist directory, and the image in the spotid directory. Hence, we have double the images.

## Distribution Plan

### 1. Create Distribution Package
- [ ] Create a clean directory for distribution
- [ ] Copy entire `dist/spotid` directory into it
- [ ] Rename directory to "SpotID-[version]" (e.g., "SpotID-v1.0")
- [ ] Create README.txt with:
  ```
  SpotID - Leopard Identification System
  Version 1.0
  
  Instructions:
  1. Double-click the 'spotid' executable
  2. Wait for your default web browser to open automatically
  3. The application interface will appear in your browser
  4. Start using SpotID!
  
  Note: The first launch may take a few moments as the system initializes.
  ```

### 2. Package for Distribution
- [ ] Create ZIP archive of the distribution directory
- [ ] Name format: "SpotID-v1.0-[platform].zip" (e.g., "SpotID-v1.0-macos.zip")
- [ ] Test the ZIP by extracting and running on a fresh system

### 3. Distribution Channels
- [ ] Create GitHub release
  - Upload ZIP file
  - Add release notes
  - Include basic usage instructions
- [ ] Update project README.md with:
  - Download links
  - Installation instructions
  - Platform requirements
  - Quick start guide

### 4. User Documentation
- [ ] Create quick start guide
- [ ] Add troubleshooting section
- [ ] Include example usage workflow
- [ ] Add system requirements
