
struct ObsFactorTest


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
    #Loop over observed factors
    for k = 1:size(obsfac, 2)
        #Get gammahat (least squares of estfac on observed factor)
        capg = obsfac[:, k]
        gammahat = estfac \ capg
        capghat = estfac * gammahat

        #Get estimated variance of least squares estfac estimate of observed factor (capg)
        varhatcapghat = (1/N) * gammahat' * invcapvtilde * capgammahat * invcapvtilde *  gammahat
    end

    #Get gamma hat
    if covmethod == :cshac || covmethod == :eq4 #Bai, Ng (2006) equation 4
        error("Use chris code here as guide. Bai Ng paper bit confusing here as they seem to
        use j as subscript for the true factors as well as for observed factors")
    elseif covmethod == :eq5 #Bai, Ng (2006) equation 5

    elseif covmethod == :eq6 #Bai, Ng (2006) equation 6
        sigsqhat = (1/(T*N)) * sum(ehat.^2)
        capgammaahat = (1/N) * sigsqhat * (facload' * facload) #[numfac*numfac]
    end




end


function testobsfactor(x::Matrix{<:Number}, obsfac::Matrix{<:Number} ; kmax::Int=10,
                       xcent::Bool=false)::NumFactor
    !xcent && (x = x .- mean(x, 1))
    numfac = numfactor(x, kmax=kmax, xcent=true)
    return testobsfactor(x, obsfac, avgnumfactor(numfac), xcent=true)
end
