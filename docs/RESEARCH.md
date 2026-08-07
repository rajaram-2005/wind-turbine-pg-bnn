# AeroVigil — Research Papers & References

> A curated collection of peer-reviewed research, technical reports, and
> resources that underpin the AeroVigil Physics-Guided Bayesian Neural Network.
> Organized by topic for easy reference.

---

## 1. Bayesian Neural Networks for RUL Prediction

| # | Paper | Authors | Venue / Year | Link |
|---|-------|---------|--------------|------|
| 1 | **Dynamic Normalized Health Indicator Construction and Bayesian Recurrent State Estimation for RUL Prediction of High-Speed Bearings in Wind Turbine Drivetrain** | X. Li, W. Teng, Y. Zhang, D. Peng, Y. Liu | Measurement, Vol. 246, 2025 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224125000843) |
| 2 | **Bayesian Estimation of Remaining Useful Life for Wind Turbine Blades** | J. S. Nielsen, J. D. Sørensen | Wind Energy, 2017 | [ResearchGate](https://www.researchgate.net/publication/353431999) |
| 3 | **Utilizing Uncertainty Information in Remaining Useful Life Estimation via Bayesian Neural Networks and Hamiltonian Monte Carlo** | — | Journal of Manufacturing Systems, 2020 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0278612520301928) |
| 4 | **Uncertainty Quantification in Multivariable Regression for Material Property Prediction with Bayesian Neural Networks** | — | Nature Scientific Reports, 2024 | [Nature](https://www.nature.com/articles/s41598-024-61189-x) |
| 5 | **Calibration of Model Uncertainty for Dropout Variational Inference** | M.-H. Laves et al. | ICML Workshop, 2020 | [ResearchGate](https://www.researchgate.net/publication/342377529) |
| 6 | **Deep Evidential Transformer with Monte Carlo Dropout for Uncertainty-Aware RUL Prediction** | — | Computers & Industrial Engineering, 2026 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360835226001804) |
| 7 | **Remaining Useful Life Prediction of Rolling Bearings Based on Bayesian Neural Network and Uncertainty Quantification** | G.-J. Jiang et al. | Quality & Reliability Engineering Int., 2023 | [Wiley](https://doi.org/10.1002/qre.3308) |

### Key Takeaways for AeroVigil
- Bayesian approaches provide **epistemic + aleatoric uncertainty decomposition** — critical for high-consequence maintenance decisions.
- **Monte Carlo Variational Inference (MCVI)** is computationally efficient compared to HMC while retaining good calibration.
- Dropout-based variational inference can be **miscalibrated** — temperature scaling is needed (Laves et al. 2020).

---

## 2. Physics-Informed Neural Networks (PINNs) for Prognostics

| # | Paper | Authors | Venue / Year | Link |
|---|-------|---------|--------------|------|
| 8 | **Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries** | P. Wen, Z.-S. Ye, Y. Li, S. Chen, P. Xie, S. Zhao | IEEE Trans. on Industrial Electronics, 2023 | [arXiv](https://arxiv.org/abs/2301.00776) |
| 9 | **Wind Turbine Main Bearing Fatigue Life Estimation with Physics-Informed Neural Networks** | — | ResearchGate, 2019 | [ResearchGate](https://www.researchgate.net/publication/335727899) |
| 10 | **A Physics-Informed Neural Network Framework for Big Machinery Data in PHM** | S. Cofre Martel | PhD Dissertation, U. Maryland, 2022 | [DRUM](https://drum.lib.umd.edu/items/845ecfba-786b-40e8-b0a7-06c0b2ebb7b0) |
| 11 | **Remaining Useful Life Prediction Based on Physics-Informed Data Augmentation** | — | Reliability Engineering & System Safety, 2024 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0951832024005234) |
| 12 | **From Physics to Machine Learning and Back: Part I — Learning with Inductive Biases in PHM** | — | Reliability Engineering & System Safety, 2026 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0951832026000293) |
| 13 | **Data-Efficient and Uncertainty-Aware RUL Prediction Using Physics-Informed Neural Network with Uncertainty Quantification** | — | PHM Society Conference, 2025 | [PHM Papers](https://papers.phmsociety.org/index.php/phmconf/article/view/4356) |
| 14 | **Remaining Useful Life Prediction of Wind Turbine Gearbox Bearings with Limited Samples Based on Prior Knowledge and PI-LSTM** | Z. Wang, P. Gao, X. Chu | Sustainability, Vol. 14, 2022 | [MDPI](https://www.mdpi.com/2071-1050/14/19/12094) |

### Key Takeaways for AeroVigil
- **Physics-informed loss functions** (Weibull-based, ISO 281-based) prevent predictions from drifting into physically impossible regimes.
- **Hybrid models** (physics + data-driven) consistently outperform pure ML or pure physics approaches.
- **PINN frameworks** enable embedding of PDEs, ODEs, or algebraic constraints as soft or hard regularizers.

---

## 3. Wind Turbine Bearing Failure Prediction

| # | Paper | Authors | Venue / Year | Link |
|---|-------|---------|--------------|------|
| 15 | **Remaining Useful Life Prediction of Wind Turbine Main-Bearing Based on TSA-LSTM** | — | IEEE Access, 2024 | [IEEE](https://ieeexplore.ieee.org/iel8/7361/10577556/10552197.pdf) |
| 16 | **Prognosis of a Wind Turbine Gearbox Bearing Using Supervised Machine Learning** | — | Applied Sciences (PMC), 2019 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679281/) |
| 17 | **An Ensemble Learning Solution for Predictive Maintenance of Wind Turbines Main Bearing** | — | Applied Sciences (PMC), 2021 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926535/) |
| 18 | **Prognosis of Wind Turbine Gearbox Bearing Failures Using SCADA and Modeled Data** | L. Williams, A. Desai, Y. Guo, S. Sheng, C. Phillips | PHM Society Conference, 2020 | [PHM Papers](https://papers.phmsociety.org/index.php/phmconf/article/download/1292/862) |
| 19 | **SCADA Data-Driven Wind Turbine Main Bearing Fault Prognosis Based on Principal Component Analysis** | — | ResearchGate, 2022 | [ResearchGate](https://www.researchgate.net/publication/361069562) |
| 20 | **Health Assessment and RUL Prediction of Wind Turbine HSSBs** | — | Energies, Vol. 14, 2021 | [PDF](https://psecommunity.org/wp-content/plugins/wtor/includes/file/2303/LAPSE-2023.25567-1v1.pdf) |
| 21 | **Remaining Useful Life Prediction of Wind Turbine Gearbox Bearings with Limited Samples Based on Prior Knowledge and PI-LSTM** | Z. Wang, P. Gao, X. Chu | Sustainability, 2022 | [MDPI](https://www.mdpi.com/2071-1050/14/19/12094) |

### Key Takeaways for AeroVigil
- **SCADA data alone** is sufficient for meaningful degradation tracking — no dedicated vibration sensors required.
- **Hybrid approaches** (SCADA + physics-based modeled data) reduce false alarms by **~50%** and improve F1 by **~12%** (NREL study).
- **45-day lead time** is the industry-accepted minimum for planned maintenance intervention.

---

## 4. Monte Carlo Variational Inference & Uncertainty

| # | Paper / Resource | Authors | Venue / Year | Link |
|---|-------|---------|--------------|------|
| 22 | **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning** | Y. Gal, Z. Ghahramani | ICML, 2016 | [PDF](http://proceedings.mlr.press/v48/gal16.pdf) |
| 23 | **Variational Dropout and the Local Reparameterization Trick** | D. P. Kingma et al. | NeurIPS, 2015 | [arXiv](https://arxiv.org/abs/1506.02557) |
| 24 | **Weight Uncertainty in Neural Networks (Bayesian Backpropagation)** | C. Blundell et al. | ICML, 2015 | [arXiv](https://arxiv.org/abs/1505.05424) |
| 25 | **Multi-Level Monte Carlo Dropout for Efficient Uncertainty Quantification** | — | arXiv, 2026 | [arXiv](https://arxiv.org/pdf/2601.13272) |
| 26 | **Accelerating Hamiltonian Monte Carlo for Bayesian Inference in Neural Networks** | — | CMAME, 2025 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782525006735) |

### Key Takeaways for AeroVigil
- **MC Dropout** (Gal & Ghahramani 2016) is the theoretical foundation for AeroVigil's inference approach — keeping dropout active at test time gives approximate Bayesian inference.
- **Variational inference** provides a scalable alternative to MCMC while maintaining calibrated uncertainty estimates.
- **Epistemic vs aleatoric** decomposition is critical: epistemic (model) uncertainty decreases with more data; aleatoric (sensor) uncertainty is irreducible.

---

## 5. Wind Turbine SCADA Data & Condition Monitoring

| # | Paper / Resource | Authors | Venue / Year | Link |
|---|-------|---------|--------------|------|
| 27 | **SCADA Data for Wind Turbine Data-Driven Condition/Performance Monitoring: A Review** | — | ResearchGate, 2022 | [ResearchGate](https://www.researchgate.net/publication/363658272) |
| 28 | **Fault Detection of a Wind Turbine Generator Bearing Using Interpretable Machine Learning** | — | Frontiers in Energy Research, 2023 | [Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2023.1284676/full) |
| 29 | **Weibull-Neural Network Framework for Wind Turbine Lifetime Monitoring and Disturbance Identification** | — | Wind Energy (Wiley), 2026 | [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/we.70103) |
| 30 | **Machine Learning for Gearbox Fault Prediction by Using Both SCADA and Modeled Data** | L. Williams et al. (NREL) | NREL Technical Report, 2021 | [OSTI](https://www.osti.gov/biblio/1769826) |
| 31 | **Automatic Fault Prediction of Wind Turbine Main Bearing Based on ANN** | — | SCIRP, 2017 | [SCIRP](https://www.scirp.org/journal/paperinforcitation?paperid=85657) |

---

## 6. Industry Standards & Norms

| Standard | Title | Scope | Relevance to AeroVigil |
|----------|-------|-------|------------------------|
| **ISO 281:2007** | Rolling bearings — Dynamic load ratings and rating life | Defines L10 bearing life calculation | **Core physics constraint** in PG-BNN loss function |
| **ISO 10816-3** | Mechanical vibration — Evaluation of machine vibration | Zone boundaries for vibration severity | Vibration limit thresholds in digital twin |
| **ISO 13373-1** | Condition monitoring and diagnostics of machines | Vibration monitoring guidelines | Feature extraction from SCADA vibration data |
| **IEC 61400-25** | Wind turbines — Communications for monitoring | SCADA data model standardization | Input signal definitions |
| **IEC 61508** | Functional safety of electrical/electronic systems | Safety integrity levels | Advisory-only safety contract design |
| **ISO 14118** | Safety of machinery — Prevention of unintended start-up | Operational safety | Safety boundary definitions |
| **OSHA 29 CFR 1910.147** | Control of hazardous energy (LOTO) | Lockout/tagout procedures | Out of scope — advisory only |

---

## 7. Key Textbooks & Foundational References

| # | Book / Resource | Author(s) | Year | Notes |
|---|----------------|-----------|------|-------|
| 32 | **Deep Learning** | I. Goodfellow, Y. Bengio, A. Courville | 2016 | Bayesian neural networks (Ch. 8), dropout regularization |
| 33 | **Pattern Recognition and Machine Learning** | C. M. Bishop | 2006 | Bayesian inference, variational methods |
| 34 | **Probabilistic Machine Learning: An Introduction** | K. P. Murphy | 2022 | Modern Bayesian deep learning, MC methods |
| 35 | **Reliability Engineering and Risk Analysis** | M. Modarres, M. Kaminskiy, V. Krivtsov | 2016 | Weibull analysis, bearing life models |
| 36 | **Wind Turbine Technology: Fundamental Concepts of Wind Turbine Operation** | S. Schwartz et al. (NREL) | 2011 | Drivetrain architecture, bearing configurations |
| 37 | **Rolling Bearing Analysis** | T. A. Harris, M. N. Kotzalas | 2006 | ISO 281 derivations, load-life relationships |

---

## 8. Open Datasets for Wind Turbine Prognostics

| Dataset | Description | Source | Link |
|---------|-------------|--------|------|
| **IMS / PRONOSTIA** | Run-to-failure bearing vibration data | IEEE PHM 2012 Challenge | [FEMTO-ST](https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-challenge) |
| **C-MAPSS (NASA)** | Turbofan engine degradation simulation | NASA Ames | [NASA](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |
| **NREL 5MW Reference** | Offshore wind turbine reference model | NREL | [NREL](https://www.nrel.gov/wind/nrel-reference-turbines.html) |
| **DTU 10MW Reference** | 10MW reference wind turbine | DTU Wind Energy | [DTU](https://dtuwinder.gitlab.io/references/10mw/) |
| **OC3 Hywind** | NREL offshore 5MW spar-buoy | NREL/OC3 | [NREL](https://www.nrel.gov/wind/hywind.html) |
| **PHM Data Challenge (2022)** | Wind turbine SCADA data for fault detection | PHM Society | [PHM](https://www.phmsociety.org/phm-data-challenge/) |

---

## 9. Blog Posts & Community Resources

| # | Title | Source | Link |
|---|-------|--------|------|
| 38 | **What Uncertainties Do We Need in Bayesian Deep Learning?** | Y. Gal (Blog) | [Blog](http://yakofi.com/uncertainties/) |
| 39 | **Physics-Informed Neural Networks: A Primer** | Towards Data Science | [Medium](https://towardsdatascience.com/solving-pdes-using-deep-learning-97d3c60a2935) |
| 40 | **Wind Turbine Main Bearing Failures — A Review** | Wind Systems Magazine | [WindSystems](https://windsystemsmag.com/) |
| 41 | **Understanding Bayesian Neural Networks** | Distill.pub | [Distill](https://distill.pub/2019/visual-exploration-gaussian-processes/) |
| 42 | **Global Wind Report 2024** | Global Wind Energy Council (GWEC) | [GWEC](https://gwec.net/) |

---

## 10. Related Open-Source Projects

| Project | Description | Link |
|---------|-------------|------|
| **Pyro (Uber)** | Deep universal probabilistic programming with PyTorch | [GitHub](https://github.com/pyro-ppl/pyro) |
| **Blitz (torchbayesian)** | Bayesian neural networks for PyTorch | [GitHub](https://github.com/piEsposito/blitz-bayesian-deep-learning) |
| **OpenPHM** | Open-source prognostics and health management toolkit | [GitHub](https://github.com/wkdzwd/open-phm) |
| **WTPFM** | Wind turbine prognostics and failure management | Research codebases |

---

*This bibliography is maintained as a living document. Contributions welcome via [GitHub Discussions](https://github.com/rajaram-2005/wind-turbine-pg-bnn/discussions).*

*Last updated: August 2026*
