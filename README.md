# 581FinalProject
Final Project for PHYS 581.


This repo contains optimizers (conjugate gradient method, gradient decent, and BFGS) with line searches (Armijo and Stong Wolfe conditions) for use in nonlinear data fitting.

We use pixi for version control, and mystmd to run a documentation server. 


To open the doc server, first run

```{code-block} python
pixi shell
```

then run the following code:

```{code-block} python
myst start --execute
```

In the data folder, I have provided example "experimental data" of damped harmonic motion. The run_fit.py file can be ran to arrive to an optimal solution, using BFGS with Strong Wolfe by default.

To run the script, in a pixi shell run:

```{code-block} python
python run_fit.py
```

This will use the default optimizer to find the best parameters to model the dampened harmonic oscillator, as well as upload fitted points to a .csv file in the data folder.