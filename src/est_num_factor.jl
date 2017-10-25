
"""
    NumFactor

Output type from the numfactor function. It contains the fields pcp1:Int, pcp2::Int,
pcp3::Int, icp1::Int, icp2::Int, icp3::Int, which correspond to the number of factors
recommended by the information criteria in Bai, Ng (2002)
"""
struct NumFactor
    pcp1::Int ; pcp2::Int ; pcp3::Int ; icp1::Int ; icp2::Int ; icp3::Int
    blerg
end
NumFactor() = NumFactor(-1, -1, -1, -1, -1, -1)
function Base.show(io::IO, x::NumFactor)
    println(io, "Estimated number of factors:")
    println(io, "    pcp1 = $(x.pcp1)")
    println(io, "    pcp2 = $(x.pcp2)")
    println(io, "    pcp3 = $(x.pcp3)")
    println(io, "    icp1 = $(x.icp1)")
    println(io, "    icp2 = $(x.icp2)")
    println(io, "    icp3 = $(x.icp3)")
end

"""
    numfactor(x::Matrix{<:Number} ; kwargs...)::NumFactor \n

Estimate the number of common factors in dataset x, using the method proposed
in Bai, Ng (2002) Determining the Number of Factors in Approximate Factor Models. \n

The rows in x should correspond to observations, and columns correspond to variables.
See ?NumFactor for more information on the output type. \n

This function accepts the following keyword arguments: \n
    - kmax::Int=10 <- The maximum number of common factors to test up to. Note that
    the pcp criterion tend to over-estimate the number of common factors if kmax is
    set too large, so it is not advised to set this to an arbitrarily large value. \n
    - covmatmethod::Symbol=:auto <- If x is a TxJ matrix, then set to :cols to use
    a JxJ covariance matrix, :rows for a TxT covariance matrix, or :auto to use
    whichever is smaller of JxJ and TxT. From a statistical perspective, the choice
    is irrelevant, so :auto is recommended. \n
    - covmatfunc::Function::numfactor_cov <- The function to use to calculate
    the covariance matrix of x. Most users will want the default value of this input.
    If you do use your own function, note the covariance matrix should be unscaled, and
    that your function must accept x as the first input, and covmatmethod as the
    second input \n
    - xcent::Bool=false <- Set to true if the data in x is already centred. \n
"""
function numfactor(x::Matrix{T} ; kmax::Int=10, covmatmethod::Symbol=:auto,
                   covmatfunc::Function=numfactor_cov,
                   xcent::Bool=false)::NumFactor where {T<:Number}
    (size(x, 1) < 2 || size(x, 2) < 2) && return NumFactor()
    !xcent && (x = x .- mean(x, 1))
    !any(covmatmethod .== [:auto, :rows, :cols]) && error("covmatmethod must be set to :auto, :cols, or :rows. Current value is invalid: $(covmatmethod)")
    if covmatmethod == :auto
        size(x, 1) > size(x, 2) ? (covmatmethod = :cols) : (covmatmethod = :rows)
    end
    xcov = covmatfunc(x, covmatmethod)

    error("up to here")

end
function numfactor_cov(x::Matrix{T}, covmatmethod::Symbol) where {T<:Number}
    covmatmethod == :cols && return (x'*x)
    covmatmethod == :rows && return (x*x')
    error("")
    error("covmatmethod must be set to :cols, or :rows. Current value is invalid: $(covmatmethod)")
end
