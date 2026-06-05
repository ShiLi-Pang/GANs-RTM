# Deep Learning Preconditioned Methods Based on GANs for Frequency-Domain Reverse Time Migration (FRTM)

This repository provides a **reproducible implementation** of a GAN-based preconditioning framework for solving large-scale linear systems arising in **frequency-domain reverse time migration (FRTM)**.

The method integrates a **GANs** with the **BiCGSTAB iterative solver** to construct learned preconditioners for accelerating the solution of the Helmholtz system.

---
## 1.🔍 Overview

We propose a **generative adversarial network (GAN)-based preconditioning framework**, where a CNN-based generator–discriminator architecture is trained to learn effective preconditioners for the BiCGSTAB solver. The learned preconditioner is then embedded into the iterative solution of the Helmholtz system arising in FRTM.

Compared with conventional approaches, the proposed method:

* Improves convergence speed of BiCGSTAB
* Reduces the number of iterations
* Enhances computational efficiency in frequency-domain RTM

---
## 2. 📂 Project Structure

```
GAN-fd-RTM/
│
├── GAN_fd_RTM.py              # Main framework for FRTM + GAN preconditioner
├── bicgstab.py               # BiCGSTAB solver implementation
├── generator.py              # GAN generator network
├── discriminator.py          # GAN discriminator network
├── fun_t_filter.py          # Frequency-domain filtering utilities
├── ofd_matrix_4.py          # Helmholtz system matrix construction
│
├── GANtrain/
│   └── sunken_example.ipynb # Training example notebook
│
├── Result/
│   ├── BP2004_I_mat.npy
│   ├── fielddata_I_mat.npy
│   ├── sigsbee2B_I_mat.npy
│   └── sunken_I_mat.npy
│
├── image/
│   ├── bp2004_migration_image.png
│   ├── fielddata_migration_image.png
│   ├── sigsbee2B_migration_image.png
│   └── sunken_migration_image.png
│
├── velocity model.zip        # Velocity models for testing
└── README.md
```
---
## 3. 🧱 Installation

### 3.1 Requirements
* Python ≥ 3.8
* yTorch ≥ 1.10
* NumPy
* SciPy
* Matplotlib
### 3.2 Install dependencies

```pip install numpy scipy matplotlib torch```

---

## 4. 💾 Data Preparation

Unzip the velocity models:

```unzip velocity\ model.zip```

Supported models:
* Sunken model
* Sigsbee2B model
* BP2004 benchmark model
* Field data model

---

## 5. ▶️ Running the Code
### 5.1 Training GAN Preconditioner

Open and run:

```sunken_example.ipynb```

This notebook includes:
* Training data generation
* GAN training loop
* Validation on Helmholtz system

### 5.2 Forward + Migration (Main Experiment)

Run frequency-domain RTM with GAN preconditioner:

```python GAN_fd_RTM.py```

This will:
* Build Helmholtz system
* Load velocity model
* Apply GAN-based preconditioner
* Solve using BiCGSTAB
* Output migrated image

---
## 6. ⚙️ Algorithm Summary

* Construct frequency-domain Helmholtz operator C
* Generate training samples from physical model
* Train conditional GAN to learn preconditioner
* Embed generator into BiCGSTAB solver
* Solve Cu=s iteratively with learned preconditioning
* Perform imaging condition to obtain RTM result

---
## 7. ⚠️ Reproducibility Notes

To ensure reproducibility:
* All experiments use fixed random seeds
* Velocity models are included in velocity model.zip
* Default hyperparameters are consistent with paper settings

---
## 8. ✍️ Citation

If you use this code, please cite:
Deep learning preconditioned methods based on generative adversarial networks for frequency-domain reverse time migration.

---
## 9. 🧷 Contact

For questions or issues, please contact the authors.

---
