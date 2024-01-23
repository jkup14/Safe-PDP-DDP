# Pontraygin Differentiable Programming for Tolerant Barrier State Embedded Differential Dynamic Programming
Python and Jax implementation of Pontraygin Differentiable programming for tuning parameters of the Tolerant Barrier State, embedded into a system solved by Differential Dynamic Programming (DDP).

**DIFFERENTIABLE AND TOLERANT BARRIER STATES FOR IMPROVED EXPLORATION OF SAFETY-EMBEDDED DIFFERENTIAL DYNAMIC
PROGRAMMING WITH CHANCE CONSTRAINTS**
https://drive.google.com/file/d/1zhwCyFVEwOKXpD7ElAoC9UXvtxkhy882/view?usp=sharing

Joshua Kuperman (jkup14@gmail.com, https://www.linkedin.com/in/joshuakuperman/)

Last Update 12/28/2023

Example files **pdp_double_integrator.py, pdp_diff_drive.py, pdp_quad.py** gives a general idea on how the code should work for the deterministic case, i.e. using PDP to autotune T-DBaS parameters, assuming safety and dynamics are deterministic.

(CLEAN UP IN PROGRESS, NOT CURRENTLY WORKING)Example files**pdp_double_integrator_stochastic.py, pdp_diff_drive_stochastic.py, pdp_quad_stochastic.py** provide uses of using PDP to tune T-DBaS parameters to avoid stochastic obstacles, or a safety function with mean and variance at every time step.

bas_functions: required classes to embed barrier state(s) into a dynamics function, gp_safety_function needs to be cleaned up

cost: required quadratic cost functions, stochastic_barrier_cost needs to be cleaned up

ddp_algorithsm: pdp, ddp, and helpers

system_dynamics: cart-pole, differential drive, single/double integrator, quadrotor, and unicyle implementations with a multi-agent wrapper dynamics function

plottings: some visualization functions for both 2d and 3d cases with various constraints