# QuantumPhaseClassifier

Neural network-based classifier that predicts the thermal phase of 2D Ising model spin configurations (e.g., below/above/near critical temperature)

A machine learning pipeline for classifying the thermal phase of 2D Ising model spin configurations using neural network architectures:
• MLP (Multilayer Perceptron)
• CNN (Convolutional Neural Network)
• Vision Transformer (ViT)

## References

- Carrasquilla, J., & Melko, R. G. (2017). ["Machine learning phases of matter"](https://www.nature.com/articles/nphys4035). *Nature Physics, 13, 431–434.*
- Kara, O., & Yanik, M. E. (2021). ["Predicting order parameters in 2D Ising models using Vision Transformers"](https://arxiv.org/abs/2109.13925). *arXiv:2109.13925.*
- Civitcioglu, C. (2020). ["Transfer Learning Across Lattice Types Using Convolutional Neural Networks"](https://warwick.ac.uk/fac/sci/physics/staff/academic/roemer/publications/Thesis_Civitcioglu_2020.pdf). PhD Thesis, Warwick.
- Geirhos, R., et al. (2020). ["Shortcut Learning in Deep Neural Networks"](https://arxiv.org/abs/2004.07780). *arXiv:2004.07780.*
- [statmechsims.com](https://www.statmechsims.com/models) — Open source Ising simulation image data.

The project benchmarks these models on synthetic lattice data to explore their effectiveness at detecting phase transitions.

Dataset from Kaggle:

```py
import kagglehub

#Download latest version
path = kagglehub.dataset_download("swarnainece/ising-lattice-images")
```

1. Install requirements:

```bash
#!/bin/bash
pip install torch torchvision matplotlib pandas scikit-learn
```    

2. Train the MLP model: Trains model, plots test accuracy per epoch, and saves model checkpoint to results/.

```bash
cd src
python train.py
```

3. Evaluate the model: Loads checkpoint, prints test accuracy, and shows confusion matrix.

```bash
python eval.py
```
