# Technical Context: SpotID

## Technology Stack

### Core Dependencies
```
environment.yml key components:
- python 3.11+
- pytorch
- flask
- networkx
- opencv-python
- rembg
- numpy
- waitress
```

### Development Tools
- VSCode for development
- Git for version control
- pytest for testing
- GitHub Actions for CI

## Implementation Details

### 1. Deep Learning Framework
- PyTorch for model implementation
- EfficientNetV2-B2 backbone
- Custom CosFace implementation
- CUDA support for GPU acceleration

### 2. Preprocessing Tools
- YOLO for bounding box detection
- rembg for background removal
- OpenCV for edge detection
- Custom image augmentation pipeline

### 3. Web Interface
- Flask web framework
- NetworkX for graph database
- Waitress for production server
- JSON for data persistence
- Static file serving for images

## Configuration

### Model Configuration (config.json)
```json
{
    "train_data_dir": "Path to training data",
    "test_data_dir": "Path to test data",
    "mask_only": false,
    "method": "cosface",
    "number_embedding_dimensions": 1028,
    "resize_width": 512,
    "resize_height": 256,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "cuda",
    "backbone_model": "tf_efficientnetv2_b2",
    "margin": 0.28
}
```

### Inference Configuration (config_inference.json)
```json
{
    "output_folder": "Path for embeddings",
    "unprocessed_image_folder": "Raw images path",
    "crop_output_folder": "Cropped images path",
    "bg_removed_output_folder": "Background removed path",
    "base_binary_output_folder": "Edge detected path"
}
```

## Technical Constraints

### Hardware Requirements
- Minimum 8GB RAM for preprocessing
- CUDA-capable GPU recommended for training
- CPU-only inference supported
- Storage for embeddings database

### Performance Bounds
- Preprocessing: ~10-15s per image
- Model inference: ~15s for 500 images
- Interface response: <1s for matches

### Dataset Limitations
- Average 6.4 images per leopard flank
- High variance in image quality
- Imbalanced class distribution
- Limited training data (~8900 images)

## Development Guidelines

### Code Organization
```
leopard_id/
├── dataloader/      # Data loading and augmentation
├── engine/          # Training loop implementation
├── losses/          # Loss function implementations
├── metrics/         # Evaluation metrics
├── model/          # Model architecture
├── scripts_preprocessing/  # Preprocessing scripts
└── visualization/   # Visualization tools
```

### Testing Strategy
1. Unit tests for model components
2. Integration tests for pipeline
3. Test coverage requirements
4. Performance benchmarks

### Deployment Process
1. Environment setup via conda
2. Model weights distribution
3. Interface deployment
4. Database initialization

### Maintenance Procedures
1. Regular model retraining
2. Database backups
3. Performance monitoring
4. Error logging and tracking

## Integration Points

### Data Input
- Raw image directories
- Individual leopard metadata
- Configuration files

### System Output
- Embedding vectors
- Match predictions
- Organized leopard directories
- Match reports (CSV)

### External Systems
- Camera trap systems
- Image storage systems
- Backup systems
- Monitoring tools
