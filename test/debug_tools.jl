using CommonFactorModelStats
using Distributions

function issymmetric_approx(x::Matrix{Float64}, tol::Float64=sqrt(eps()))::Bool
    size(x, 1) != size(x, 2) && error("Input is not square: $(size(x))")
    for i = 1:size(x, 1)
        for j = 1:size(x, 2)
            abs(x[i, j] - x[j, i]) > tol && error("Symmetry test failed at index ($(i), $(j)): $(x[i,j] - x[j,i])")
        end
    end
    return true
end

function sim_mvnormal(numobs::Int, muvec::Vector{<:Number}, stdvec::Vector{<:Number}, rhomat::Matrix{<:Number})::Matrix{Float64}
    numobs < 1 && error("Invalid numobs: $(numobs)")
    !(length(muvec) == length(stdvec) == size(rhomat, 1) == size(rhomat, 2)) && error("Length mismatch: $(length(muvec)), $(length(stdvec)), $(size(rhomat, 1)), $(size(rhomat, 2))")
    any(diag(rhomat) .!= 1) && error("rhomat diagonal not equal to one: $(diag(rhomat))")
    any(abs.(rhomat) .> 1) && error("rhomat contains elements greater than one (in absolute value): $(rhomat)")
    !issymmetric_approx(rhomat) && error("rhomat is not symmetric: $(rhomat)")
    covmat = [ rhomat[i,j] * stdvec[i] * stdvec[j] for i = 1:size(rhomat, 1), j = 1:size(rhomat, 2) ]
    !issymmetric_approx(covmat) && error("Logic fail: $(covmat)")
    mvdist = MvNormal(Float64.(muvec), Matrix{Float64}(Symmetric(Float64.(covmat))))
    x = rand(mvdist, numobs)
    return x'
end
function sim_common_factors(numobs::Int, numfac::Int)::Matrix{Float64}
    muvec = randn(numfac)
    stdvec = 2*rand(numfac)
    stdvec[stdvec .< 0.5] = 0.5
    rhomat = 2*rand(numfac, numfac) - 1
    rhomat = Matrix{Float64}(Symmetric(rhomat))
    for i = 1:numfac
        for j = 1:numfac
            i == j ? (rhomat[i,j] = 1.0) : (rhomat[i,j] = 0.2*rhomat[i,j])
        end
    end
    facmat = sim_mvnormal(numobs, muvec, stdvec, rhomat)
    return facmat
end
sim_factor_loadings(numfac::Int, numvar::Int, muscalar::Number)::Matrix{Float64} = muscalar + randn(numfac, numvar)
function sim_common_factor_model(numobs::Int, numfac::Int, numvar::Int, loadmu::Number, errorstd::Number)
    errorstd <= 0.0 && error("Invalid errorstd: $(errorstd)")
    emat = errorstd * randn(numobs, numvar)
    facmat = sim_common_factors(numobs, numfac)
    loadmat = sim_factor_loadings(numfac, numvar, loadmu)
    x = facmat * loadmat + emat
    return (x, facmat, loadmat, emat)
end
function sim_obs_factor(facmat::Matrix{Float64}, mu1::Number, std1::Number)
    paramvec = randn(size(facmat, 2))
    facobsexact = facmat * paramvec
    facobs = mu1 + facobsexact + std1*randn(size(facobsexact, 1), 1)
    return (facobsexact, facobs)
end


function test_num_factor(numtest::Int, numobs::Int, numvar::Int, loadmu::Number, errorstd::Number)
    for t = 1:numtest
        numfac = rand(1:8)
        (x, f, l, e) = sim_common_factor_model(numobs, numfac, numvar, loadmu, errorstd)
        nf = CommonFactorModelStats.numfactor(x, covmatmethod=:auto)
        println("($(t)) True num fac = $(numfac). Estimates = $(CommonFactorModelStats.numfactor(nf)). Average estimate = $(CommonFactorModelStats.avgnumfactor(nf))")
    end
end

function test_obs_factor(numtest::Int, numobs::Int, numvar::Int, loadmu::Number, errorstd::Number,
                        obsbias::Number, obsstd::Float64, covmethod::Symbol=:eq4, alpha::Float64=0.05)
    for t = 1:numtest
        #numfac = rand(1:8)
        numfac = 1
        (x, f, l, e) = sim_common_factor_model(numobs, numfac, numvar, loadmu, errorstd)
        (facexact, facobs) = sim_obs_factor(f, obsbias, obsstd)
        oft = CommonFactorModelStats.testobsfactor(x, facobs, numfac, xcent=false, covmethod=covmethod, alpha=alpha)
        println("($(t)) num fac = $(numfac). aj = $(oft.aj[1]), mj = $(oft.mj[1]), mjpval = $(oft.mjpval[1]), r2 = $(oft.r2[1])")
        println("-------------------------------------------")
    end
end

# numobs = 200
# numfac = 3
# numvar = 100
# loadmu = 0
# errorstd = 2
# (x, f, l, e) = sim_common_factor_model(numobs, numfac, numvar, loadmu, errorstd)

# numtest = 10;
# numobs = 100;
# numvar = 200;
# loadmu = 0;
# errorstd = 2;
# test_num_factor(numtest, numobs, numvar, loadmu, errorstd)


numtest = 4;
numobs = 1000;
numvar = 1000;
loadmu = 0;
errorstd = 5.0;
obsbias = 0.0;
obsstd = 0.1;
#obsstd = 0.0;
covmethod = :eq4;
alpha = 0.05;
test_obs_factor(numtest, numobs, numvar, loadmu, errorstd, obsbias, obsstd, covmethod, alpha)
#blah
