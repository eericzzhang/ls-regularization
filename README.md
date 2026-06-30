# lˢ-Regularized Linear Models: Model Selection Consistency (s ∈ (0,1))

This repository contains the simulation study materials used in Qidi Peng, Eric Zhang, and Tong Zhen's paper, titled "On Model Selection Consistency of ls-regularized Linear Models with s ∈ (0, 1).", including

* original R code rendition of the simulation study program
* Python-adapted rendition of the simulation study program
* seeded code and figures for repeatability


### **Citation**
As this is a supplementary simulation study for our paper, to cite this project, cite the published paper (when it is hopefully published, at least). \
Peng, Q., Zhang, E., & Zeng, T. (2026). On Model Selection Consistency of lˢ-regularized Linear Models with s ∈ (0,1). 
Preprint submitted to Electronic Journal of Statistics.

### **Overview**
This repository contains the R and Python iterations of the simulation study from Peng, Zhang & Zeng (2026). It provides the core lˢ coordinate descent solver (Algorithm 2), a cross-validation tuning routine for the penalty parameters s and λ, and three reproducible simulation examples comparing the lˢ-regularized model against the Lasso. The code supports the paper's central claim that lˢ regularization achieves variable selection consistency without requiring the Strong Irrepresentable Condition.

### **Repository Structure**
>ls-regularization\
├── seededfiguresPY &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Python output figures\
│   ├── *.png  (×8)\
│   └── figures_combined.pdf\
├── seededfiguresR       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # R output figures\
│   ├── *.png  (×8)\
│   └── figures_combined.pdf\
├── PYgen2026 code.py\
├── Rgen2026 code.R\
└── README.md

### **Requirements to Run**
For R:
- RStudio 2026.05.1+225 "Golden Wattle" Release (89e0c7da7b7b806fc5701ecb567f993716d8cfdc, 2026-06-07) for windows\
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) RStudio/2026.05.1+225\
Chrome/146.0.7680.216 Electron/41.5.0 Safari/537.36, Quarto 1.9.37

&nbsp;&nbsp;&nbsp;&nbsp;Required packages:
&nbsp;&nbsp;&nbsp;&nbsp;  - glmnet (version 5.0)

&nbsp;&nbsp;&nbsp;&nbsp;Install with:
&nbsp;&nbsp;&nbsp;&nbsp;  ```install.packages("glmnet")```

For Python: 
- Python version 3.8 or higher

&nbsp;&nbsp;&nbsp;&nbsp;Required packages:
&nbsp;&nbsp;&nbsp;&nbsp; -  numpy, matplotlib, scikit-learn

&nbsp;&nbsp;&nbsp;&nbsp;Install with:
&nbsp;&nbsp;&nbsp;&nbsp;   ```pip install numpy matplotlib scikit-learn```

### **How to Run Simulation Study in R**
Begin by opening the attached Rgen2026 code from the GitHub repository, then install and load the glmnet package

```
install.packages("glmnet") 
library(glmnet)
```

Load the GLM and tune_s_lambda functions

The GLM function is an R-adapted version of Algorithm 2 in our paper

```
GLM = function (X,Y,s,lambda,beta,nstep,a)
{n = length(Y);
d = ncol(X);
for (j in 1:d){
  mu = X[,j]%*%X[,j];
  c = t(X[,j])%*%(Y-X[,-j]%*%beta[-j,])/mu;
  lambda_critical = mu*abs(c)/(2-s)*(abs(2-2*s)/(2-s)*abs(c))^(1-s);
  if (lambda_critical <= lambda) {beta[j,]=0}
  else {
    g_first = function(x){mu*(x - c) + lambda*s*sign(x)*abs(x)^(s-1)};
    g_second = function(x){mu + lambda*s*(s-1)*abs(x)^(s-2)};
    x = c;
    for (i in 1:nstep){e = a*g_first(x)/g_second(x);
    if (e < 10^(-5)){break} else {x = x-e}};
    beta[j,] = x;
  }
}
beta0 = mean(Y-X%*%beta);
c(beta0,beta);
}
```

Tune the parameters via the test MSE

```
tune_s_lambda = function(X,Y,S,L,beta,nstep,a) {
  R = matrix(rep(10^8,S*L),ncol = L);
  Z = cbind(rep(1,nrow(X)),X);
  train = sample(1:nrow(X),nrow(X)*3/4,replace = FALSE);
  for (s in (1:S)/(S+1)){
    for (l in (1:10*L)/L){
      b = GLM(X[train,],Y[train],s,l,beta,nstep,a);
      b = matrix(b,ncol=1);
      R[s,l] = mean((Y[-train]-Z[-train,]%*%b)^2);
    }
  };
  min = which(R == min(R), arr.ind = TRUE);
  # Print best s, lambda and relational error
  c(min[1,1]/(S+1),min[1,2]/L);
}
```

Load the setup for the example for which you want to run

**Example 1.1**
```
# Example 1.1: variable selection (lasso VS l^s-regularization), the strong irrepresentable condition fails, non-variable selection consistency
# The figure shows the estimation performance of each coefficient with respect to sample size.
# Take beta1 = 2, beta2 = 3, s and lambda are optimized by using the function "tune_s_lambda".
beta1= matrix(c(2,3),ncol=1);

# initial guess of beta.
beta = rbind(beta1,0);
nstep = 100;
S=L=100;
a =1;


# Generate data X and Y
# The sample size is from n_start to n_end.
n_start = 20;
n_end = 200;
N=1:(n_end-n_start+1);

# For l^s regularization
bhat=matrix(0,n_end-n_start+1,4);

# For lasso
bhat_lasso=matrix(0,n_end-n_start+1,4);

# Generate Y = X*beta+ epsilon
for (n in n_start:n_end){
  set.seed(1);
  X1 = matrix(c(rnorm(n,0,1),rnorm(n,0,1)),nrow=n,ncol=2);
  X2 = t((2/3)%*%(X1[,1]+X1[,2]));
  X=cbind(X1,X2);
  C12=(1/n)*(t(X1)%*%X2);
  C11=(1/n)*(t(X1)%*%X1);
  C21=(1/n)*(t(X2)%*%X1);
  # Check the strong irrepresentable condition
  SIC=abs((C21%*%solve(C11))%*%sign(beta1));
  set.seed(2);
  e = matrix(rnorm(n,0,1),nrow=n,ncol=1);
  Y = X%*%beta+e;
  
  
  # Run Lasso: fit with cross-validation
  set.seed(3);
  cv_model = cv.glmnet(X, Y, alpha = 1);
  best_lambda = cv_model$lambda.min;
  best_model = glmnet(X, Y, alpha = 1, lambda = best_lambda);
  out_lasso = coef(best_model);
  for (k in 1:4){bhat_lasso[n-n_start+1,k] = out_lasso[k]};
  
  # Run l^s regularization
  # Tune s and lambda
  set.seed(4);
  tune_parameter=tune_s_lambda(X,Y,S,L,beta,nstep,a);
  s=tune_parameter[1];
  lambda=tune_parameter[2];
  out = GLM(X,Y,s,lambda,beta,nstep,a);
  for (k in 1:4){bhat[n-n_start+1,k] = out[k]};
}
```

**Example 1.2**

```
# Example 1.2: variable selection (lasso VS l^s-regularization), the strong irrepresentable condition holds.
# The figure shows the estimation performance of each coefficient with respect to sample size.
# Take beta1 = -2, beta2 = 3, s and lambda are optimized by using the function "tune_s_lambda".
beta1= matrix(c(-2,3),ncol=1);

# initial guess of beta.
beta = rbind(beta1,0);
nstep = 100;
S=L=100;
a =1;

# Generate data X and Y
# The sample size is from n_start to n_end.
n_start = 20;
n_end = 200;

# For l^s regularization
bhat=matrix(0,n_end-n_start+1,4);

# For lasso
bhat_lasso=matrix(0,n_end-n_start+1,4);

# Generate Y = X*beta+ epsilon
for (n in n_start:n_end){
  set.seed(1);
  X1 = matrix(c(rnorm(n,0,1),rnorm(n,0,1)),nrow=n,ncol=2);
  X2 = t((2/3)%*%(X1[,1]+X1[,2]));
  X=cbind(X1,X2);
  C12=(1/n)*(t(X1)%*%X2);
  C11=(1/n)*(t(X1)%*%X1);
  C21=(1/n)*(t(X2)%*%X1);
  # Check the strong irrepresentable condition
  SIC=abs((C21%*%solve(C11))%*%sign(beta1));
  set.seed(2);
  e = matrix(rnorm(n,0,1),nrow=n,ncol=1);
  Y = X%*%beta+e;
  
  
  # Run Lasso: fit with cross-validation
  set.seed(3)
  cv_model = cv.glmnet(X, Y, alpha = 1);
  best_lambda = cv_model$lambda.min;
  best_model = glmnet(X, Y, alpha = 1, lambda = best_lambda);
  out_lasso = coef(best_model);
  for (k in 1:4){bhat_lasso[n-n_start+1,k] = out_lasso[k]};
  
  # Run l^s regularization
  # Tune s and lambda
  set.seed(4)
  tune_parameter=tune_s_lambda(X,Y,S,L,beta,nstep,a);
  s=tune_parameter[1];
  lambda=tune_parameter[2];
  out = GLM(X,Y,s,lambda,beta,nstep,a);
  for (k in 1:4){bhat[n-n_start+1,k] = out[k]};
}



```

**Example 2.1**

```
# Example 2.1: Estimates VS norm of beta via lambda: the strong irrepresentable condition fails.
# We take beta1=2, beta2=3, beta3=0.

beta1= matrix(c(2,3),ncol=1);

# initial guess of beta.
beta = rbind(beta1,0);
nstep = 100;
a =1;

# Generate data X and Y
# sample size
n=1000;


# For l^s regularization
bhat=matrix(0,n,4);
beta_norm = rep(0,n);
# For lasso
bhat_lasso=matrix(0,n,4);
beta_norm_lasso = rep(0,n);


# Generate X, Y.
set.seed(1);
X1 = matrix(c(rnorm(n,0,1),rnorm(n,0,1)),nrow=n,ncol=2);
X2 = t((2/3)%*%(X1[,1]+X1[,2]));
X=cbind(X1,X2);
C12=(1/n)*(t(X1)%*%X2);
C11=(1/n)*(t(X1)%*%X1);
#IC=abs(solve(C11)%*%C12);
C21=(1/n)*(t(X2)%*%X1);
SIC=abs((C21%*%solve(C11))%*%sign(beta1));
set.seed(2);
e = matrix(rnorm(n,0,1),nrow=n,ncol=1);
Y = X%*%beta+e;

# Fix s
s = 0.9;

for (m in 1:n){
  # Run Lasso
  lambda = 0.001*m;
  best_model = glmnet(X, Y, alpha = 1, lambda = lambda);
  out_lasso = coef(best_model);
  # Run l^s
  lambda = exp(m);
  out = GLM(X,Y,s,lambda,beta,nstep,a);
  beta_norm[m] = (sum(abs(out[2:4])^s))^{1/s};
  beta_norm_lasso[m] = sum(abs(out_lasso[2:4]));
  for (j in 1:4){
    bhat[m,j] = out[j];
    bhat_lasso[m,j] = out_lasso[j];
  }
}

beta_norm_order = order(beta_norm);
beta_norm_new = beta_norm[beta_norm_order];
beta_norm_order_lasso = order(beta_norm_lasso);
beta_norm_lasso_new = beta_norm_lasso[beta_norm_order_lasso];
for (j in 1:4){
  bhat[,j] = bhat[beta_norm_order,j];
  bhat_lasso[,j] = bhat_lasso[beta_norm_order_lasso,j];}
```

**Example 2.2**
```
# Example 2.2: Estimates VS norm of beta via lambda: the strong irrepresentable condition holds.
# We take beta1=-2, beta2=3, beta3=0.

beta1= matrix(c(-2,3),ncol=1);

# initial guess of beta.
beta = rbind(beta1,0);
nstep = 100;
a =1;

# Generate data X and Y
# sample size
n=1000;


# For l^s regularization
bhat=matrix(0,n,4);
beta_norm = rep(0,n);
# For lasso
bhat_lasso=matrix(0,n,4);
beta_norm_lasso = rep(0,n);


# Generate X, Y.
set.seed(1);
X1 = matrix(c(rnorm(n,0,1),rnorm(n,0,1)),nrow=n,ncol=2);
X2 = t((2/3)%*%(X1[,1]+X1[,2]));
X=cbind(X1,X2);
C12=(1/n)*(t(X1)%*%X2);
C11=(1/n)*(t(X1)%*%X1);
#IC=abs(solve(C11)%*%C12);
C21=(1/n)*(t(X2)%*%X1);
SIC=abs((C21%*%solve(C11))%*%sign(beta1));
set.seed(2);
e = matrix(rnorm(n,0,1),nrow=n,ncol=1);
Y = X%*%beta+e;

# Fix s
s = 0.9;

for (m in 1:n){
  # Run Lasso
  lambda = 0.001*m;
  best_model = glmnet(X, Y, alpha = 1, lambda = lambda);
  out_lasso = coef(best_model);
  # Run l^s
  lambda = exp(m);
  out = GLM(X,Y,s,lambda,beta,nstep,a);
  beta_norm[m] = (sum(abs(out[2:4])^s))^{1/s};
  beta_norm_lasso[m] = sum(abs(out_lasso[2:4]));
  for (j in 1:4){
    bhat[m,j] = out[j];
    bhat_lasso[m,j] = out_lasso[j];
  }
}

beta_norm_order = order(beta_norm);
beta_norm_new = beta_norm[beta_norm_order];
beta_norm_order_lasso = order(beta_norm_lasso);
beta_norm_lasso_new = beta_norm_lasso[beta_norm_order_lasso];
for (j in 1:4){
  bhat[,j] = bhat[beta_norm_order,j];
  bhat_lasso[,j] = bhat_lasso[beta_norm_order_lasso,j];}

```

> Note that each randomizer in each example has been seeded for the purposes of experimental repeatability.\
 Seeds $1$ through $4$ are used in examples 1.1 and 1.2, while only seeds $1$ and $2$ are used in examples 2.1 and 2.2. 

To plot the figures for each example, run the setup for said example first, then run the plotting code for your desired method (lasso or lˢ) beneath each example

E.g., Example 1.1, figure 1.1.1

```
# Figure 1.1.1: for lasso solution path of beta 4 suggests no variable selection consistency

# Plot lasso
plot(N,bhat_lasso[,4],ylim=c(-2,4), main=expression("Estimates of "*beta[i]*" from the Lasso"), xlab="Sample Size n", ylab=expression(beta[i]), type="p", lty=2, col="red");
points(N,bhat_lasso[,2],type="l", lty=2, col="blue");
points(N,bhat_lasso[,3],type="l", lty=2, col="black");
abline(h=2,col="blue");
abline(h=0,col="red");
abline(h=3,col="black");
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("blue","black","red"))
SIC;
```

### **How to Run Simulation Study in Python**
Begin by loading the required packages

```
pip install numpy matplotlib scikit-learn 
```

The entire script is self-contained. Running it in any terminal will create all 8 figures saved as .png files to the directory. As figures display interactively as the script runs, you must close each plot window manually before the script can continue to the next step. 

> Each randomizer in this script is seeded in the same way as in the R rendition, with Seeds $1$ through $4$ in examples 1.1 and 1.2, and seeds $1$ and $2$ used in examples 2.1 and 2.2. \
Note that even though both scripts use the same seeds, their results can still vary, as they do here. 


### **Simulation Study Results**
Inside the [seeded figuresPY](seededfiguresPY) and [seeded figuresR](seededfiguresR) folders contains all of our simulation study results in the form of .pngs and a combined pdf for ease of viewing for the Python and R renditions respectivley. 


## **Contacts**
Qidi Peng\
Institute of Mathematical Sciences, Claremont Graduate University\
Claremont, CA 91711, U.S.A.\
qidi.peng@cgu.edu

Eric Zhang\
College of Letters and Science, University of California, Los Angeles\
Los Angeles, CA 90095, U.S.A.\
eericzhang@g.ucla.edu

Tong Zeng\
Department of Business Sciences and Economics, University of La Verne\
La Verne, CA 91750, U.S.A.\
tzeng@laverne.edu
