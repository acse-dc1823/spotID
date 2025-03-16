# System Patterns: SpotID

## Architecture Overview

### Core Components
```mermaid
graph TD
    A[Raw Images] --> B[Preprocessing Pipeline]
    B --> C[Deep Learning Model]
    C --> D[Embedding Database]
    D --> E[Interface]
    E --> F[Match Database]
```

## Design Patterns

### 1. Preprocessing Pipeline
```mermaid
graph LR
    A[Input Image] --> B[YOLO Detection]
    B --> C[Bounding Box]
    C --> D[Background Removal]
    D --> E1[RGB Channels]
    D --> E2[Edge Detection]
    E1 --> F[Combined Channels]
    E2 --> F
```

### 2. Model Architecture
```mermaid
graph TD
    A[Input Image] --> B[EfficientNetV2-B2]
    B --> C[FC Layer]
    C --> D[1028D Embedding]
    D --> E[Angular Space]
    E --> F[Modified CosFace]
```

### 3. Data Flow Patterns

#### Training Flow
- Dataset organization by individual/flank
- Custom sampler for batch creation
- Semi-hard negative mining
- Adaptive angular margin calculation

#### Inference Flow
- Preprocessing of new images
- Embedding generation
- Cosine similarity comparison
- Top-5 match selection

### 4. Interface Patterns

#### User Interaction Flow
```mermaid
graph TD
    A[Load Images] --> B[Generate Embeddings]
    B --> C[Compare Patterns]
    C --> D[User Verification]
    D --> E[Graph Database Update]
    E --> F[Export Results]
```

## Key Technical Decisions

### 1. Model Selection
- **Decision**: Use EfficientNetV2-B2 backbone
- **Rationale**: Balance between performance and computational efficiency
- **Consequences**: Better accuracy than ResNet-18 with similar parameter count

### 2. Angular Margin Function
- **Decision**: Implement modified margin function h(θ)
- **Rationale**: Better handling of hard exemplars and minority classes
- **Consequence**: Improved separation in embedding space

### 3. Database Architecture
- **Decision**: Use graph database for matches
- **Rationale**: Natural representation of leopard relationships
- **Consequence**: Efficient querying and update operations

### 4. Interface Design
- **Decision**: Flask-based web interface
- **Rationale**: Lightweight, easy to deploy
- **Consequence**: Accessible to researchers without installation

## Performance Patterns

### Optimization Strategies
1. Image preprocessing caching
2. Batch processing for embeddings
3. Efficient graph traversal for matches
4. Memory-efficient storage of embeddings

### Error Handling
1. Graceful degradation for failed preprocessing
2. User verification for uncertain matches
3. Progress saving for long sessions
4. Automatic backup of match database

## Testing Patterns
1. Unit tests for model components
2. Integration tests for pipeline
3. Accuracy metrics on test set
4. Performance benchmarking
