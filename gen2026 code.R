install.packages("glmnet")
library(glmnet)

# Solution to the L^s linear regularized model (Algorithm 2 in Peng, Zhang and Tong (2026))
# X = designed matrix
# Y = response
# s = order of the penalty. 0<s<1.
# lambda = tuning parameter. Degree of tolerance of the variance.
# beta = initial value of the coefficients.
# a = scaling parameter of the Newton's method.

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

# Tuning s and lambda in the L^s linear regularization
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

# Figure 1.1.2: plot l^s regularization

plot(N,bhat[,4],ylim=c(-2,4), main=expression("Estimates of "*beta[i]*" from the "*l^s*"-regularization"), xlab="Sample Size n", ylab=expression(beta[i]), type="p", lty=2, col="red");
points(N,bhat[,2],type="l", lty=2, col="blue");
points(N,bhat[,3],type="l", lty=2, col="black");
abline(h=2,col="blue");
abline(h=0,col="red");
abline(h=3,col="black");
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("blue","black","red"))
SIC;

####################################################################################################

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

# Figure 1.2.1: for lasso  solution path of beta 4 suggests no variable selection consistency

# Plot lasso
plot(N,bhat_lasso[,4],ylim=c(-6,4), main=expression("Estimates of "*beta[i]*" from the Lasso"), xlab="Sample Size n", ylab=expression(beta[i]), type="p", lty=2, col="red");
points(N,bhat_lasso[,2],type="l", lty=2, col="blue");
points(N,bhat_lasso[,3],type="l", lty=2, col="black");
abline(h=-2,col="blue");
abline(h=0,col="red");
abline(h=3,col="black");
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("blue","black","red"))
SIC;

# Figure 1.2.2: plot l^s regularization

plot(N,bhat[,4],ylim=c(-6,4), main=expression("Estimates of "*beta[i]*" from the "*l^s*"-regularization"), xlab="Sample Size n", ylab=expression(beta[i]), type="p", lty=2, col="red");
points(N,bhat[,2],type="l", lty=2, col="blue");
points(N,bhat[,3],type="l", lty=2, col="black");
abline(h=-2,col="blue");
abline(h=0,col="red");
abline(h=3,col="black");
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("blue","black","red"))
SIC;


#####################################################################################################

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

# Figure 2.1.1: solution path of beta 4 suggests no variable selection consistency
# L^s model
plot(beta_norm_new,bhat[,4],main=expression("Solution Paths of "*L^s*" - Linear Regularization against "*"\u2016"*beta*"\u2016"[s]), ylim=c(-3.5,3.5), xlab=expression("\u2016"*beta*"\u2016"[s]),
     ylab=expression("Estimate of "*beta[i]), type="l", lty=2, col="blue");
points(beta_norm_new,bhat[,2], type="l", lty=2, col="red");
points(beta_norm_new,bhat[,3], type="l", lty=2, col="black");
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("red","black","blue"))
#############################################################################################

# Figure 2.1.2
# Lasso 
plot(beta_norm_lasso_new,bhat_lasso[,4],main=expression("Solution Paths of Lasso against "*"\u2016"*beta*"\u2016"[1]), ylim=c(-3.5,3.5), xlab=expression("\u2016"*beta*"\u2016"[1]),
     ylab=expression("Estimate of "*beta[i]), type="l", lty=2, col="blue");
points(beta_norm_lasso_new,bhat_lasso[,2], type="l", lty=2, col="red");
points(beta_norm_lasso_new,bhat_lasso[,3], type="l", lty=2, col="black");

#############################################################################################
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("red","black","blue"))


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

# Figure 2.2.1: solution path of beta 4 suggests no variable selection consistency
# L^s model
plot(beta_norm_new,bhat[,4],main=expression("Solution Paths of "*L^s*" - Linear Regularization against "*"\u2016"*beta*"\u2016"[s]), ylim=c(-3.5,3.5), xlab=expression("\u2016"*beta*"\u2016"[s]),
     ylab=expression("Estimate of "*beta[i]), type="l", lty=2, col="blue");
points(beta_norm_new,bhat[,2], type="l", lty=2, col="red");
points(beta_norm_new,bhat[,3], type="l", lty=2, col="black");
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("red","black","blue"))
#############################################################################################

# Figure 2.2.2
# Lasso 
plot(beta_norm_lasso_new,bhat_lasso[,4],main=expression("Solution Paths of Lasso against "*"\u2016"*beta*"\u2016"[1]), ylim=c(-3.5,3.5), xlab=expression("\u2016"*beta*"\u2016"[1]),
     ylab=expression("Estimate of "*beta[i]), type="l", lty=2, col="blue");
points(beta_norm_lasso_new,bhat_lasso[,2], type="l", lty=2, col="red");
points(beta_norm_lasso_new,bhat_lasso[,3], type="l", lty=2, col="black");

#############################################################################################
# Adding a legend
legend("bottomleft", legend=c(expression(beta[1]),expression(beta[2]),expression(beta[3])), fill=c("red","black","blue"))

