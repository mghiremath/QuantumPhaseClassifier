# QuantumPhaseClassifier

Neural network-based classifier that predicts the thermal phase of 2D Ising model spin configurations (e.g., below/above/near critical temperature)

A machine learning pipeline for classifying the thermal phase of 2D Ising model spin configurations using neural network architectures:
• MLP (Multilayer Perceptron)
• CNN (Convolutional Neural Network)
• Vision Transformer (ViT)

The project benchmarks these models on synthetic lattice data to explore their effectiveness at detecting phase transitions.

Dataset from Kaggle:

```py
import kagglehub

#Download latest version
path = kagglehub.dataset_download("swarnainece/ising-lattice-images")
```

    1.	Install requirements:

    ```bash
    #!/bin/bash
    pip install torch torchvision matplotlib pandas scikit-learn
    ```

    2.	Train the MLP model: Trains model, plots test accuracy per epoch, and saves model checkpoint to results/.

    ```bash
    cd src
    python train.py
    ```

    3.	Evaluate the model: Loads checkpoint, prints test accuracy, and shows confusion matrix.

    ```bash
    python eval.py
    ```
