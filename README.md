# Hankel-Alternative-View-of-Koopman-HAVOK-analysis


A Python implementation of the HAVOK analysis.

This work was part of my Masters thesis submitted at the University of Hamburg, Germany. Please feel free to refer to my thesis.

The HAVOK model can be used for qualitative prediction of highly nonlinear and chaotic systems. To summarise, the HAVOK analysis consists of the following steps:

1. Takens' embedding theorem, Hankel matrix and the SVD - One of the most important results in differential topology is the Takens' embedding theorem. It is used to reconstruct a strange attractor (in this case, the Lorenz Attractor) using only a single variable (the x-coordinate in this case). As shown by Broomhead in 1986, in his method of delays, the right singular values from the singular value decomposition (SVD) of a time delayed Hankel matrix of a single variable not only satisfy Takens' embedding theorem, but provide a better reconstruction than using purely time delayed data points of a single variable. The first three right singular vectors form the reconstructed attractor.

![image](https://user-images.githubusercontent.com/49671867/215022833-ecfef7f7-3b8e-4ddd-a4ad-87a3c3178665.png)

2. Koopman Operator theory and the intermittently forced linear system - Koopman operator theory tells us that finite dimensional nonlinear systems can be thought of as an infinite dimensional linear operator progressing the initial value through time. We then seek to approximate a finite dimensional version of the infinite dimensional Koopman operator. The first 'r' right singular vectors from the SVD of the time delayed Hankel matrix form an approximately invariant space, and since these are almost linearly independent, we can not fit a linear system, but fit an intermittently forced linear system. We use the first 'r' right singular vectors to fit the intermittently forced linear system, with the rth right singular vector as the forcing term.

![image](https://user-images.githubusercontent.com/49671867/215022947-1dd5c016-14aa-474a-8b6e-b8d43086f942.png)


The first step transforms our system using only partial variable data, into a topologically equivalent system, where the qualitative dynamics of the original system are preserved.

The second step fits an intermittently forced linear system to the topologically equivalent system and using this, we can make qualitative predictions.

In the original HAVOK implementation, the least energy right singular value was used as the forcing term. The idea of using multiple forcing terms was acknowledged, but not explored. In my thesis, the use of higher energy forcing terms along with multiple forcing terms was explored. It was shown that when two forcing terms were used, we only needed a total of five right singular values. To get a similar result with only a single forcing term, we needed 15 right singular values. This suggests that when using multiple forcing terms, the amount of data required to train the intermittently forced linear system drastically reduces. This can reduce time taken to run the model for systems with huge datasets, and can be run on smaller processers, for example on sensors.

Further work that can be done:
1. Optimising the number of forcing terms. What happens when we use 3 or more forcing terms?
2. Instead of Takens' embedding theorem, we can also use Deyle and Sugihara's theorem for state space reconstructions (SSR) where multiple variables are used for reconstructions instead of the single variable SSR as in Takens' embedding theorem.
3. Hybrid ML-HAVOK model that is closed, meaning it can give future predictions instead of only decomposing a nonlinear system to a linear one.
4. A real time HAVOK model.

References:

- Steven L. Brunton et al. “Chaos as an intermittently forced linear system”. In: Nature Communications 8.1 (May 2017).
- David S Broomhead, Gregory P King, et al. “On the qualitative analysis of experimental dynamical systems”. In: Nonlinear phenomena and chaos 113 (1986), p. 114.
- Steven L Brunton, Joshua L Proctor, and J Nathan Kutz. “Discovering governing equations from data by sparse identification of nonlinear dynamical systems”. In: Proceedings of the national academy of sciences 113.15 (2016), pp. 3932–3937.

