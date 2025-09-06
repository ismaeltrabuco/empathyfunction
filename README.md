# Empathy Function and Empathetic - PINN

Empathy Function teaches machine empathy to understand context in feeling and emotional signals from inputs

> *"Empathy, I believe, must become code."*

A first prototype of **Empathetic Physics-Informed Neural Networks (E-PINNs)**.
Born from the fusion of mathematics, physics, fashion, and performance art.

##  What The Empathy Function Does

The Empathy Function measures **dynamic alignment between a system’s internal state and external feedback**, enabling AI systems to not just predict, but to *tune themselves* in ways coherent with human-like behavior.

### Core Innovation

Our implementation is inspired by operator theory and empathy models such as Niko Sauer’s work:

```
y_{t+1} = y_t + λ · cov(y_t, o_t)
```

Where:

* **y\_t**: current system state
* **o\_t**: external observations
* **λ**: sensitivity factor

This equation works as the “heart” of the Empathetic Physics-Informed Neural Network (E-PINN).

##  Why This Project

Traditional ML often struggles with:

* Black-box predictions with low interpretability
* Ignoring physical and behavioral constraints
* Lack of coherence between internal states and observed outcomes

Our approach:

*  **Physics-Informed** (conservation, entropy, latency laws)
*  **Human-Interpretable** (outputs like `3&24` = 3/5 likelihood, class 24)
*  **Creative** (bridging human sciences + computational modeling)

##  Getting Started

Currently this is a **research prototype**.
Code lives in this repo in two main modules:

* `empathy_function/` → empathy function implementation
* `epinn/` → experimental E-PINN model

Example (toy dataset):

```python
from empathy_function import EmpathyScorer
from epinn import EPINNModel

# Sample data
data = {
    'visits': [980, 1020, 880, 1100, 970],
    'stories': [3, 4, 2, 5, 3],
    'clicks': [200, 220, 190, 250, 205],
    'moon_phase': [2, 3, 4, 0, 1],
    'sales': [38, 42, 33, 48, 37]
}

# Calculate empathy scores
empathy_scorer = EmpathyScorer()
scores = empathy_scorer.calculate_empathy(data)

# Train E-PINN model
model = EPINNModel(physics_constraints=['attention_conservation'])
model.fit(data, scores)

print("Visitor IDs:", model.generate_visitor_ids())
```

##  Current Architecture

```
Input → Empathy Function → Physics-Informed Training → Output
```

* **Empathy Function**: covariance-based update
* **Physics Constraints**: attention conservation, diminishing returns, latency
* **Output**: Propensity & Class ID (`3&24`)

##  Context & Research

This project is part of:

* Master’s thesis in **Art Education & Performance Art** (UNESP)
* Exploration of **Empathy as Code** in human-computer interfaces
* Early step toward a PhD-level inquiry in **Data Modeling**

##  Contributing

We invite collaborators interested in:

* Additional physics-informed constraints
* Visualization of empathy scores
* Testing on real-world datasets
* Open discussions on empathy in computational systems

Contact: [@ismaeltrabuco on Telegram](https://t.me/ismaeltrabuco)

##  Maintainer

* **Ismael Trabuco** – Creator & Researcher

  * Master’s student at UNESP (Art Education & Performance Art)
  * Thesis: *“Spectra-Classroom: The Empathic Interface for a Performance Art of Emotional Mathematics”*

---

*"Empathy must become code — not to replace human empathy, but to bridge it with computational intelligence."*

