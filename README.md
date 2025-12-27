# Speaker Recognition Project using x-vectors

This repository contains an end-to-end implementation of a Speaker Recognition system using x-vectors (Time-Delay Neural Networks) on the VoxCeleb1 dataset. The project is divided into two distinct implementations based on dataset usage.

## implementations

### 1. Prototype Implementation (10% Subset)
Located in the root directory, this implementation was designed as a proof-of-concept using a 10% subset of the available data (approximately 24 speakers).

*   **Location**: Root directory (`D:\x_vectors_dec2025`)
*   **Purpose**: Initial validation of the pipeline and model architecture.
*   **Dataset**: 10% subset of VoxCeleb1 Dev set.

### 2. Full Dataset Implementation
Located in the `x_vectors_for_complete_VoxCeleb1_dataset` directory, this implementation utilizes the complete available dataset (240 speakers found in the provided path).

*   **Location**: `x_vectors_for_complete_VoxCeleb1_dataset/`
*   **Purpose**: Production-ready training on the full dataset.
*   **Dataset**: Complete VoxCeleb1 Dev set (as provided).

## Project Structure

```text
x_vectors_dec2025/
├── src/                        # Source code for Prototype Implementation
│   ├── dataset.py              # Data loading and processing
│   ├── features.py             # MFCC feature extraction
│   ├── model.py                # TDNN x-vector model architecture
│   ├── train.py                # Training loop implementation
│   ├── plda.py                 # Backend (LDA/PLDA) implementation
│   ├── evaluate.py             # Evaluation metrics (EER)
│   └── extract_vectors.py      # Embedding extraction logic
├── main.py                     # Entry point for Prototype Implementation
├── requirements.txt            # Project dependencies
├── x_vectors_for_complete_VoxCeleb1_dataset/
│   ├── src/                    # Source code for Full Dataset Implementation
│   ├── main.py                 # Entry point for Full Dataset Implementation
│   └── requirements.txt        # Dependencies for Full Dataset Implementation
└── README.md                   # Project documentation
```

## Prerequisites

*   **Python**: 3.8+
*   **CUDA Toolkit**: Required for GPU acceleration (Verified on RTX 3050).
*   **Dataset**: VoxCeleb1 dataset located at `G:\3 x vector with 10 percent dataset\Dataset`.

## Installation

1.  Create and activate the virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Full Dataset Implementation

1.  Navigate to the full dataset project directory:
    ```bash
    cd x_vectors_for_complete_VoxCeleb1_dataset
    ```

2.  **Training**:
    To train the model (default: 20 epochs):
    ```bash
    ..\venv\Scripts\python main.py train
    ```
    Model checkpoints are saved in the `checkpoints/` directory.

3.  **Verification**:
    To extract embeddings and evaluate the Equal Error Rate (EER):
    ```bash
    ..\venv\Scripts\python main.py extract
    ```

### Running the Prototype (10% Subset)

1.  Navigate to the root directory:
    ```bash
    cd D:\x_vectors_dec2025
    ```

2.  **Training**:
    ```bash
    .\venv\Scripts\python main.py train
    ```

3.  **Verification**:
    ```bash
    .\venv\Scripts\python main.py extract
    ```

## Results

| Implementation | Speakers | Samples | Training Accuracy | Validation EER |
| :--- | :--- | :--- | :--- | :--- |
| **Prototype (10%)** | 24 | ~5,185 | 93.65% | 10.00% |
| **Full Dataset** | 240 | ~12,281 | 92.35% | 8.44% |

## Methodology

*   **Features**: 24-dimensional MFCCs with Cepstral Mean Normalization (CMN).
*   **Architecture**: standard x-vector TDNN topology with Statistics Pooling.
*   **Loss Function**: Cross-Entropy Loss.
*   **Backend**: Cosine Similarity scoring on length-normalized embeddings.
