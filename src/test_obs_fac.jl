

"""
    ObsFactorTest

Output type from the testobsfactor function. The fields of this type include: \n
    -aj: Test statistic defined in Bai, Ng (2006) equation 2. aj <= alpha implies
        the corresponding observed factor is close to spanning the latent factor
        space. aj --> 1 implies the corresponding observed factor is unlikely
        to span the latent factor space.
    -mj: Test statistic defined in Bai, Ng (2006) equation x. This test statistic
        is used to test the null hypothesis that the observed factor is an exact
        linear combination of the latent factor space.
    -mjpval: The p-value associated with the mj statistic.
    -r2: A heuristic metric of how well the observed factor fits the latent factor
        space. r2 --> 0 implies a very poor fit, r2 --> 1 implies a very good fit.
        Note that, heuristically, the aj and mj statistics tend to be very sensitive,
        ie observed factors that deviate from the latent factor space by a small
        amount of random noise can have aj very close to 1 and mjpval very close to 0.
        In contrast the r2 measure tends to be more stable.
"""
struct ObsFactorTest #aj, mj, mjpval, r2, all have one element for each observed factor
    aj::Vector{Float64}
    mj::Vector{Float64}
    mjpval::Vector{Float64}
    r2::Vector{Float64}
end

"""
    testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number} ; kwargs) \n
    testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number}, numfac::Int ; kwargs) \n

The function tests whether the observated factors in obsfac span the same space as the latent
factor space that is estimated from input dataset x. The testing methodology is described in
Bai, Ng (2006) Evaluating Latent and Observed Factors in Macroeconomics and Finance [Journal of
Econometrics]. \n

The number of factors is estimated using Bai, Ng (2002), or else the user can specify the number
of factors using a third argument. \n

Keyword arguments are as follows: \n
    - xcent::Bool=false <-- Set to true if x is already centred \n
    - covmethod::Symbol=:cshac <-- Choose the method for computing the asymptotic variance, corresponding
        to equations 4, 5, and 5 in the original paper. Use either (:eq4, :cshac) for equation 4,
        either (:eq5, :het) for equation 5, or either (:eq6, :iid) for equation 6. \n
    - alpha::Float64=0.05 <-- The significance level to use when computing statistics \n
    - standardize::Bool=true <-- Set to true to standardize x and obsfac to have unit variance \n

See ?ObsFactorTest for more information on the output type, and the meaning of the various
statistics it contains. \n
"""
function testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number}, numfac::Int ;
                       xcent::Bool=false, covmethod::Symbol=:cshac, alpha::Float64=0.05,
                       standardize::Bool=true)::ObsFactorTest
    #Preliminaries
    (T, N) = (size(x, 1), size(x, 2))
    N < numfac && return ObsFactorTest()
    (T < 2 || N < 2) && return ObsFactorTest()
    !xcent && (x = x .- mean(x, 1))
    standardize && (x = x .* (1 ./ std(x, 1)))
    #Get covariance matrix
    xcov = (1 / (T*N)) * x * x'
    #Get eigenvalues and eigenvectors for largest numfac eigenvalues, and then get factor loadings and estimated factors
    ef = eigfact(Symmetric(xcov), size(xcov,1)-numfac+1:size(xcov,1))
    (eigvalvec, eigvecmat) = (flipdim(ef.values, 1), flipdim(ef.vectors, 2))
    estfac = sqrt(T) * eigvecmat #\tilde{F} from Bai, Ng (2006)  [T*numfac]
    facload = (1/T) * (x' * estfac) #\tilde{Lambda} from Bai, Ng, (2006) [N*numfac]
    ehat = x - estfac*facload' #\tilde{e} from Bai, Ng (2006) [T*N]
    #Get eigenvalue inverse matrix
    invcapvtilde = diagm(1 ./ eigvalvec) #[numfac*numfac]
    #Get gamma hat
    if covmethod == :cshac || covmethod == :eq4 #Bai, Ng (2006) equation 4
        nupper = Int(floor(min(sqrt(T), sqrt(N))))
        capgammahat = zeros(Float64, numfac, numfac)
        for i = 1:nupper
            for j = 1:nupper
                capgammahat += (1 / nupper) * facload[i, :] * facload[j, :]' * cgh_inner_sum(ehat, i, j)
            end
        end
    elseif covmethod == :het || covmethod == :eq5 #Bai, Ng (2006) equation 5
        capgammahatvec = [ zeros(Float64, numfac, numfac) for t = 1:T ]
        for t = 1:T
            for i =  1:N
                capgammahatvec[t] += ehat[t, i]^2 * facload[i, :] * facload[i, :]'
            end
            capgammahatvec[t] *= (1/N)
        end
    elseif covmethod == :iid || covmethod == :eq6 #Bai, Ng (2006) equation 6
        sigsqhat = (1/(T*N)) * sum(ehat.^2)
        capgammahat = (1/N) * sigsqhat * (facload' * facload) #[numfac*numfac]
    end
    #Loop over observed factors
    mjvec = Vector{Float64}(size(obsfac, 2))
    ajvec = Vector{Float64}(size(obsfac, 2))
    r2vec = Vector{Float64}(size(obsfac, 2))
    for k = 1:size(obsfac, 2)
        #Get gammahat (least squares of estfac on observed factor)
        capg = obsfac[:, k] - mean(view(obsfac, :, k))
        standardize && (capg = (1 / std(capg)) * capg)
        gammahat = estfac \ capg
        capghat = estfac * gammahat
        #Get estimated variance of least squares estfac estimate of observed factor (capg)
        if any(covmethod .== [:eq5, :het]) ; varhatcapghatvec = [ (1/N) * gammahat' * invcapvtilde * capgammahat * invcapvtilde *  gammahat for capgammahat in capgammahatvec ]
        else ; varhatcapghat = (1/N) * gammahat' * invcapvtilde * capgammahat * invcapvtilde *  gammahat
        end
        #Get tau statistic
        if any(covmethod .== [:eq5, :het]) ; tau = (capghat - capg) ./ sqrt.(varhatcapghatvec)
        else ; tau = (capghat - capg) / sqrt(varhatcapghat)
        end
        abstau = abs.(tau)
        #Get mj
        mjvec[k] = maximum(abstau)
        #Get aj
        ajvec[k] = mean(abstau .> quantile(Normal(0,1), 1 - (alpha/2)))
        #Get R2
        r2vec[k] = T*dot(gammahat, gammahat)*(1/dot(capg, capg))
        #r2vec[k] = var(capghat) / var(capg)
    end
    mjvec_pval = [ 1 - (2*cdf(Normal(0,1), mj) - 1)^T for mj in mjvec ]
    return ObsFactorTest(ajvec, mjvec, mjvec_pval, r2vec)
end
cgh_inner_sum(ehat::Matrix{Float64}, i::Int, j::Int)::Float64 = (1 / size(ehat, 1)) * dot(view(ehat, :, i), view(ehat, :, j))
function testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number} ; kmax::Int=10,
                       xcent::Bool=false)::ObsFactorTest
    !xcent && (x = x .- mean(x, 1))
    numfac = numfactor(x, kmax=kmax, xcent=true)
    return testobsfactor(x, obsfac, avgnumfactor(numfac), xcent=true)
end
