# 🔍 Understanding Autograd from Scratch

This repository is a learning-focused exploration of how an automatic differentiation system like **Autograd** is implemented under the hood. The goal is to **simulate the internal workings** of Autograd by manually building a computational graph and implementing essential components of backpropagation.

## 📁 Project Structure

- `test_autograd_like.py`:  
  Contains simple unit test and usage example that validate the implemented logic of the autograd system.

## 🚧 Project Scope

> **Note:** This is not a complete or production-ready implementation. Only selected core components are implemented to aid **conceptual understanding** of how Autograd systems operate.

This includes:
- Building nodes and connecting them into a computational graph.
- Forward and backward pass logic (for basic operations).
- Manual tracking of gradients through a basic class system.

## 🎯 Goals

- Demystify how automatic differentiation frameworks like PyTorch's Autograd work internally.
- Understand how gradients are propagated through a computational graph.
- Experiment with simplified custom operations to observe how they affect the graph structure and gradient flow.

## ✨ Future Enhancements

- Add support for more operations (e.g., matrix multiplication, non-linear activations).
- Implement a better graph visualization/debugging tool.
- Extend the testing suite to handle edge cases.

---

Made with curiosity and a bit of mathematical caffeine ☕  
Pull requests and suggestions are welcome!