
struct ObsFactorTest
    aj::Vector{Float64} #One element for each observed factor
    mj::Vector{Float64} #One element for each observed factor
    r2::Vector{Float64} #One element for each observed factor
end

"""
    testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number} ; kwargs)
    testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number}, numfac::Int ; kwargs)

The function tests whether the observated factors in obsfac span the same space as the latent
factor space that is estimated from input dataset x. The testing methodology is described in
Bai, Ng (2006) Evaluating Latent and Observed Factors in Macroeconomics and Finance [Nournal of
Econometrics].
"""
function testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number}, numfac::Int ;
                       xcent::Bool=false, covmethod::Symbol=:cshac)::ObsFactorTest
    #Preliminaries
    (T, N) = (size(x, 1), size(x, 2))
    N < numfac && return ObsFactorTest()
    (T < 2 || N < 2) && return ObsFactorTest()
    !xcent && (x = x .- mean(x, 1))
    #Get covariance matrix
    xcov = (1 / (T*N)) * x * x'
    #Get eigenvalues and eigenvectors for largest numfac eigenvalues, and then get factor loadings and estimated factors
    ef = eigfact(Symmetric(xcov), size(xcov,1)-numfac+1:size(xcov,1))
    (eigvalvec, eigvecmat) = (flipdim(ef.values, 1), flipdim(ef.vectors, 2))
    estfac = sqrt(T) * eigvecmat #\tilde{F} from Bai, Ng (2006)  [T*numfac]
    facload = (1/T) * (x' * estfac) #\tilde{Lambda} from Bai, Ng, (2006) [N*numfac]
    ehat = Xcent - estfac*facload' #\tilde{e} from Bai, Ng (2006) [T*N]
    #Get eigenvalue inverse matrix
    invcapvtilde = diagm(1 ./ eigvalvec) #[numfac*numfac]
    #Get gamma hat
    if covmethod == :cshac || covmethod == :eq4 #Bai, Ng (2006) equation 4
        nupper = Int(floor(min(sqrt(T), sqrt(N))))
        #Method 1
        capgammahat = zeros(Float64, numfac, numfac)
        for t = 1:T
            for i = 1:nupper
                for j = 1:nupper
                    capgammahat += facload[i, :] * facload[j, :]' * ehat[t, i] * ehat[t, j]
                end
            end
        end
        capgammahat = (1 / (nupper*T)) * capgammahat
        #Method 2
        capgammahat = zeros(Float64, numfac, numfac)
        for i = 1:nupper
            for j = 1:nupper
                capgammahat += (1 / nupper) * facload[i, :] * facload[j, :]' * cgh_inner_sum(ehat, i, j)
            end
        end

        cgh_inner_sum(ehat::Matrix{Float64}, i::Int, j::Int)::Float64 = (1 / size(ehat, 1)) * dot(view(ehat, :, i), view(ehat, :, j))


    elseif covmethod == :eq5 #Bai, Ng (2006) equation 5
        error("This variant is not currently available. Note for this equation, capgammahat depends on t, and so subsequent computations involving this estimator are more complicated and will need their own code path")
        capgammahatvec = [ zeros(Float64, numfac, numfac) for t = 1:T ]
        for t = 1:T
            for i =  1:N
                capgammahatvec[t] += ehat[t, i]^2 * facload[i, :] * facload[i, :]'
            end
            capgammahatvec[t] *= (1/N)
        end
    elseif covmethod == :eq6 #Bai, Ng (2006) equation 6
        sigsqhat = (1/(T*N)) * sum(ehat.^2)
        capgammaahat = (1/N) * sigsqhat * (facload' * facload) #[numfac*numfac]
    end

    #Loop over observed factors
    for k = 1:size(obsfac, 2)
        #Get gammahat (least squares of estfac on observed factor)
        capg = obsfac[:, k]
        gammahat = estfac \ capg
        capghat = estfac * gammahat

        #Get estimated variance of least squares estfac estimate of observed factor (capg)
        if covmethod == :eq5 ; varhatcapghatvec = [ (1/N) * gammahat' * invcapvtilde * capgammahat * invcapvtilde *  gammahat for capgammahat in capgammahatvec ]
        else ; varhatcapghat = (1/N) * gammahat' * invcapvtilde * capgammahat * invcapvtilde *  gammahat
        end
        #Get tau statistic
        if covmethod == :eq5 ; tau = (capghat - capg) ./ sqrt.(varhatcapghatvec)
        else ; tau = (capghat - capg) / sqrt(varhatcapghat)
        end


    end





end


function testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number} ; kmax::Int=10,
                       xcent::Bool=false)::NumFactor
    !xcent && (x = x .- mean(x, 1))
    numfac = numfactor(x, kmax=kmax, xcent=true)
    return testobsfactor(x, obsfac, avgnumfactor(numfac), xcent=true)
end
