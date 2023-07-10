# Mi4-based stationary pattern detector simulation
A code to run the simulation presented in [Tanaka et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.01.04.522814v1.abstract) Fig. S5.
This notebook simulates Mi4 as a linear neuron with temporally low-pass, spatially
derivative taking neuron. Then, it simulates activity of hypothetical downstream
neurons, whose mean activity can signal the presence of stationary patterns.

## Requirements
- numpy
- scipy
- matplotlib

## References
- [Arenz et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28343964/) for Mi4 receptive field property.
- [Stavenga (2003)](https://pubmed.ncbi.nlm.nih.gov/12664095/) for Drosophila photoreceptor receptive angles.
