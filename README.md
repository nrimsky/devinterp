# DevInterp / use of SGLD to estimate learning coefficient

_Produced during [Developmental Interpretability hackathon](https://events.humanitix.com/hackathon-developmental-interpretability) by Nina Rimsky and Dmitry Vaintrob_

- Using SGLD to estimate learning coefficient $\hat{\lambda}$ for a toy model trained on modular addition (approach based on the paper [Quantifying degeneracy in singular models via the learning coefficient](https://arxiv.org/pdf/2308.12108v1.pdf)).
- Visualizing the development of Fourier modes over training, both by looking at their magnitudes in the weights and by looking at the impact of projecting to particular Fourier subspaces on the loss.
- Distinguishing ["Pizza" vs "Clock"](https://browse.arxiv.org/pdf/2306.17844.pdf) circuits by looking at the relationship between the Fourier transform of post-embed activations and pre-unembed activations. Given a large Fourier mode $k$ post-embed, the pre-unembed activations will show a large mode at $2k \text{ mod } p$ for "Pizza" circuits but only at $k$ for "Clock" circuits.

_Related Alignment Forum post_: [Investigating the learning coefficient of modular addition: hackathon project](https://www.alignmentforum.org/posts/4v3hMuKfsGatLXPgt/investigating-the-learning-coefficient-of-modular-addition)