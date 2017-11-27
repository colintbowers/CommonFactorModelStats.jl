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


function test_obs_factor_bai_ng(numtest::Int, numobs::Int, numvar::Int, covmethod::Symbol, alpha::Float64)
    ajmat = Array{Float64}(numtest, 7)
    mjmat = Array{Float64}(numtest, 7)
    mjpvalmat = Array{Float64}(numtest, 7)
    mjrejmat = Array{Float64}(numtest, 7)
    r2mat = Array{Float64}(numtest, 7)
    deltamat = [1.0 1.0 0.0 ; 1.0 0.0 0.0 ; 1.0 1.0 0.2 ; 1.0 0.0 0.2 ; 1.0 1.0 2.0 ; 1.0 0.0 2.0 ; 0.0 0.0 1.0]
    for m = 1:numtest
        mod(m, 50) == 0 && println("Up to $(m)")
        f = randn(numobs, 2)
        l = randn(2, numvar)
        resid = randn(numobs, numvar)
        x = f*l + resid
        x = std_x_mat(x)
        for d = 1:7
            gtemp = f*deltamat[d, 1:2]
            g = gtemp + deltamat[d, 3] * randn(length(gtemp))
            g = reshape(g, length(g), 1)
            tof = CommonFactorModelStats.testobsfactor(x, g, 2, covmethod=covmethod, alpha=0.05)
            ajmat[m, d] = tof.aj[1]
            mjmat[m, d] = tof.mj[1]
            mjpvalmat[m, d] = tof.mjpval[1]
            tof.mjpval[1] < 0.05 ? (mjrejmat[m, d] = 1.0) : (mjrejmat[m, d] = 0.0)
            r2mat[m, d] = tof.r2[1]
        end
    end
    println("Test result with numtest=$(numtest), numobs=$(numobs), and numvar=$(numvar)")
    println("Avg aj = $(mean(ajmat, 1))")
    println("Avg mj rej = $(mean(mjrejmat, 1))")
    println("Avg r2 = $(mean(r2mat, 1))")
    println("----------------------------")
    nothing
end
std_x_vec(x::Vector{Float64})::Vector{Float64} = (1 / std(x)) * (x - mean(x))
std_x_mat(x::Matrix{Float64})::Matrix{Float64} = hcat([ std_x_vec(x[:, k]) for k = 1:size(x, 2) ]...)

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


# numtest = 4;
# numobs = 1000;
# numvar = 1000;
# loadmu = 0;
# errorstd = 1.0;
# obsbias = 0.0;
# obsstd = 0.05;
# #obsstd = 0.0;
# covmethod = :eq4;
# alpha = 0.05;
# test_obs_factor(numtest, numobs, numvar, loadmu, errorstd, obsbias, obsstd, covmethod, alpha)
#blah


numtest = 1000;
numobs = 50;
numvar = 50;
covmethod = :eq6;
alpha = 0.05;
test_obs_factor_bai_ng(numtest, numobs, numvar, covmethod, alpha);
