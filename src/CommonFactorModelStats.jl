"""
Module for statistics for common factor models (or approximate factor models), by Colin T Bowers \n
\n
This module implements the following statistics for use with common factor models: \n
    - numfactor: Estimating the number of factors using the methods from Bai, Ng (2002) Determining the Number
    of Factors in Approximate Factor Models
    - testobsfactor: Testing whether an observed factor spans the latent common factor space using the methods
    from Bai, Ng (2006) Evaluating Latent and Observed Factors in Macroeconomics and Finance
\n
This package has an MIT license. Please see associated LICENSE.md file for more detail.
"""
module CommonFactorModelStats

using Distributions

include("est_num_factor.jl")
include("test_obs_fac.jl")

# package code goes here

end # module
