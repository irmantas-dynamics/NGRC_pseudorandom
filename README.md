New Generation Reservoir Computing using pseudorandom nonlinear projection of time-delay embedded input according to the paper:
"Next-Generation Reservoir Computing for Dynamical Inference"
https://doi.org/10.48550/arXiv.2509.11338

Two maps are tested:
* X_{n+1} = W_{out} * P(X_n,X_{n-1},...) in file Lorenz_ODE.m
* X_{n+1} = X_{n} + W_{out} * P(X_n,X_{n-1},...) in file Lorenz_ODE_dX.m
