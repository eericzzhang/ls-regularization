# lˢ-Regularized Linear Models: Model Selection Consistency (s ∈ (0,1))

**Simulation study materials for Qidi Peng, Eric Zhang, and Tong Zhen's paper, titled "On Model Selection Consistency of ls-regularized Linear Models with s ∈ (0, 1)."**

**Citation**\
As this is a supplementary simulation study for our paper, to cite this project, cite the published paper.\
Peng, Q., Zhang, E., & Zeng, T. (2026). On Model Selection Consistency of lˢ-regularized Linear Models with s ∈ (0,1). 
Preprint submitted to Electronic Journal of Statistics. (NOT PUBLISHED YET, NO FORMAL CITATION, EDIT LATER)

**Overview**\
This repository contains the R implementation of the simulation study from Peng, Zhang & Zeng (2026). It provides the core lˢ coordinate descent solver, a cross-validation tuning routine for the penalty parameters s and λ, and three reproducible simulation examples comparing the lˢ-regularized model against the Lasso. The code supports the paper's central claim that lˢ regularization achieves variable selection consistency without requiring the Strong Irrepresentable Condition.

**Repository Structure**\
(ADD THIS WHEN THE ENTIRE RESPOSITORY IS SET UP)

**Requirements to Run**\
RStudio 2026.05.1+225 "Golden Wattle" Release (89e0c7da7b7b806fc5701ecb567f993716d8cfdc, 2026-06-07) for windows\
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) RStudio/2026.05.1+225\
Chrome/146.0.7680.216 Electron/41.5.0 Safari/537.36, Quarto 1.9.37

&nbsp;&nbsp;&nbsp;&nbsp;Required packages:
&nbsp;&nbsp;&nbsp;&nbsp;  - glmnet (version 5.0)

&nbsp;&nbsp;&nbsp;&nbsp;Install with:
&nbsp;&nbsp;&nbsp;&nbsp;  install.packages("glmnet")

**How to Run**\
(ADD WHEN REPOSITORY IS FULLY ORGANIZED)

**Reproducability/Seeds**\
(ADD WHEN CODE AND GRAPHS ARE IMPLEMENTED)

**Simulation Study Examples**\

**Key Parameters**\

**What the Figures Show**\

>The Strong Irrepresentable Condition (SIC) is the requirement that |C₂₁C₁₁⁻¹ sign(β*)| ≤ 1−η for some η > 0, where C₁₁ and C₂₁ are subblocks of the normalized Gram matrix XᵀX/n. It is a necessary condition for Lasso's variable selection consistency but depends on the unknown true coefficient vector β*, making it unverifiable in practice. The lˢ-regularized model achieves variable selection consistency under conditions that depend only on the observed design matrix X.


**Contacts**\
Qidi Peng\
Institute of Mathematical Sciences, Claremont Graduate University\
Claremont, CA 91711, U.S.A.\
qidi.peng@cgu.edu

Eric Zhang\
University of California, Los Angeles\
Los Angeles, CA 90095, U.S.A.\
eericzhang@g.ucla.edu

Tong Zeng\
Department of Business Sciences and Economics, University of La Verne\
La Verne, CA 91750, U.S.A.\
tzeng@laverne.edu
