
"""
    NumFactor

Output type from the numfactor function. The fields of this type provide the
number of factors for each information criterion, as well as the underlying information
criterion vectors for each criterion. A description of the information criterion
can be found in Bai, Ng (2002) section 5. \n

The estimated number of factors can be obtained using numfactor(nf::NumFactor)
to get a vector of all 6 estimates, corresponding to criterions pc1, pc2, pc3,
ic1, ic2, ic3. Alternatively, you can get just one of these criterion using
numfactor(nf::NumFactor, y::Symbol) where y can take values :pc1, :pc2, :pc3,
:ic1, :ic2, :ic3. y::Vector{Symbol} is also accepted as input. \n

The average estimate of number of factors taken across all 6 criterion, and rounded
to the nearest integer, can be obtained using avgnumfactor(nf::NumFactor) \n
"""
struct NumFactor
    pc1numfac::Int ; pc2numfac::Int ; pc3numfac::Int ; ic1numfac::Int ; ic2numfac::Int ; ic3numfac::Int
    pc1vec::Vector{Float64} ; pc2vec::Vector{Float64} ; pc3vec::Vector{Float64}
    ic1vec::Vector{Float64} ; ic2vec::Vector{Float64} ; ic3vec::Vector{Float64}
end
NumFactor() = NumFactor(-1, -1, -1, -1, -1, -1)
numfactor(nf::NumFactor)::Vector{Int} = [nf.pc1numfac, nf.pc2numfac, nf.pc3numfac, nf.ic1numfac, nf.ic2numfac, nf.ic3numfac]
function numfactor(nf::NumFactor, icsym::Symbol)::Int
    icsym == :pc1 && return nf.pc1numfac
    icsym == :pc2 && return nf.pc2numfac
    icsym == :pc3 && return nf.pc3numfac
    icsym == :ic1 && return nf.ic1numfac
    icsym == :ic2 && return nf.ic2numfac
    icsym == :ic3 && return nf.ic3numfac
    error("Invalid information criterion symbol $(icsym). Please use :pc1, :pc2, :pc3, :ic1, :ic2, or :ic3")
end
numfactor(nf::NumFactor, icsymvec::Vector{Symbol})::Vector{Int} = [ numfactor(nf, icsym) for icsym in icsymvec ]
avgnumfactor(nf::NumFactor)::Int = Int(round(mean(numfactor(nf))))
function Base.show(io::IO, x::NumFactor)
    println(io, "Estimated number of factors:")
    println(io, "    pcp1 = $(x.pc1numfac)")
    println(io, "    pcp2 = $(x.pc2numfac)")
    println(io, "    pcp3 = $(x.pc3numfac)")
    println(io, "    icp1 = $(x.ic1numfac)")
    println(io, "    icp2 = $(x.ic2numfac)")
    println(io, "    icp3 = $(x.ic3numfac)")
end

"""
    numfactor(x::Matrix{<:Number} ; kwargs...)::NumFactor \n

Estimate the number of common factors in dataset x, using the method proposed
in Bai, Ng (2002) Determining the Number of Factors in Approximate Factor Models [Econometrica]. \n

The rows in x should correspond to observations, and columns correspond to variables.
See ?NumFactor for more information on the output type. \n

This function accepts the following keyword arguments: \n
    - kmax::Int=10 <- The maximum number of common factors to test up to. Note that
    the pcp criterion tend to over-estimate the number of common factors if kmax is
    set too large, so it is not advised to set this to an arbitrarily large value. \n
    - covmatdim::Symbol=:auto <- If x is a TxJ matrix, then set to :cols to use
    a JxJ covariance matrix, :rows for a TxT covariance matrix, or :auto to use
    whichever is smaller of J or T. From a statistical perspective, the choice
    is irrelevant, so :auto is recommended. \n
    - xcent::Bool=false <- Set to true if the data in x is already centred. \n
    - standardize::Bool=true <-- Set to true to standardize x to have unit variance \n
"""
function numfactor(x::Matrix{<:Number} ; kmax::Int=10, covmatdim::Symbol=:auto,
                   xcent::Bool=false, standardize::Bool=false)::NumFactor
    #Preliminaries
    (T, J) = (size(x, 1), size(x, 2))
    if covmatdim == :auto ; min(T, J) < kmax && return NumFactor()
    elseif covmatdim == :rows ; J < kmax && return NumFactor()
    elseif covmatdim == :cols ; T < kmax && return NumFactor()
    else ; error("Invalid covmatdim: $(covmatdim)")
    end
    (T < 2 || J < 2) && return NumFactor()
    !xcent && (x = x .- mean(x, 1))
    standardize && (x = x .* (1 ./ std(x, 1)))
    !any(covmatdim .== [:auto, :rows, :cols]) && error("covmatdim must be set to :auto, :cols, or :rows. Current value is invalid: $(covmatdim)")
    if covmatdim == :auto #Choose whichever dimension is smaller for covariance matrix
        T > J ? (covmatdim = :cols) : (covmatdim = :rows)
    end
    #Get covariance matrix
    if covmatdim == :cols ; xcov = x'*x
    elseif covmatdim == :rows ; xcov = x*x'
    else ; error("covmatdim must be set to :auto, :cols, or :rows. Current value is invalid: $(covmatdim)")
    end
    #Get eigenvalues and eigenvectors for largest kmax eigenvalues
    ef = eigfact(Symmetric(xcov), size(xcov,1)-kmax+1:size(xcov,1))
    (eigvalvec, eigvecmat) = (flipdim(ef.values, 1), flipdim(ef.vectors, 2))
    #Get estimated factors and factor loadings (see Bai, Ng (2002) section 3)
    if covmatdim == :cols
        facload = sqrt(J) * eigvecmat
        estfac = (1/J) * x * facload
    elseif covmatdim == :rows
        estfac = sqrt(T) * eigvecmat
        facload = ((1/T) * x' * estfac)
    else ; error("Logic fail. It should have been impossible to reach this point. Please file an issue.")
    end
    #Get average sum of squared residuals (see Bai, Ng (2002) equation 7)
    v = [ (1/(T*J)) * sum((x' - facload[:, 1:k] * estfac[:, 1:k]').^2) for k = 1:kmax ]
    #Set some useful values
    tj1 = (T + J) / (T * J)
    tj2 = 1 / tj1
    c2 = (min(sqrt(T), sqrt(J)))^2
    #Get PC criterion vectors
    pc1vec = v + (v[end]*tj1*log(tj2))*collect(1:kmax)
    pc2vec = v + (v[end]*tj1*log(c2))*collect(1:kmax)
    pc3vec = v + (v[end]*(log(c2)/c2))*collect(1:kmax)
    #Get IC criterion vectors
    ic1vec = log.(v) + (tj1*log(tj2))*collect(1:kmax)
    ic2vec = log.(v) + (tj1*log(c2))*collect(1:kmax)
    ic3vec = log.(v) + (log(c2)/c2)*collect(1:kmax)
    #Return a NumFactor object
    (pc1nf, pc2nf, pc3nf, ic1nf, ic2nf, ic3nf) = (indmin(pc1vec), indmin(pc2vec), indmin(pc3vec), indmin(ic1vec), indmin(ic2vec), indmin(ic3vec))
    return NumFactor(pc1nf, pc2nf, pc3nf, ic1nf, ic2nf, ic3nf, pc1vec, pc2vec, pc3vec, ic1vec, ic2vec, ic3vec)
end
