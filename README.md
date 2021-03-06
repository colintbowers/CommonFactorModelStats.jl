CommonFactorModelStats.jl
=====================

[![Build Status](https://travis-ci.org/colintbowers/CommonFactorModelStats.jl.svg?branch=master)](https://travis-ci.org/colintbowers/CommonFactorModelStats.jl)

[![Coverage Status](https://coveralls.io/repos/colintbowers/CommonFactorModelStats.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/colintbowers/CommonFactorModelStats.jl?branch=master)

[![codecov.io](http://codecov.io/github/colintbowers/CommonFactorModelStats.jl/coverage.svg?branch=master)](http://codecov.io/github/colintbowers/CommonFactorModelStats.jl?branch=master)

A module for the Julia language that implements several statistical tests from the literature on common factor models (or approximate factor models).

Bai, Ng (2002) "Determining the Number of Factors in Approximate Factor Models" *Econometrica*

Bai, Ng (2006) "Evaluating Latent and Observed Factors in Macroeconomics and Finance" *Journal of
Econometrics*

This package is in a first draft state.

For determining the number of factors, use `?numfactor` at the REPL to see the function documentation, and `?NumFactor` to see documentation on the ouput type of the function.

For evaluating observed factors use `testobsfactor` at the REPL to see the function documenation, and `?ObsFactorTest` to see documentation on the output type of the function.
