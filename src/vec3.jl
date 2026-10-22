"""
    Vec3(x, y, z)
    Vec3(v)

An immutable 3-vector used for points, directions and RGB colors.
`Vec3(v)` broadcasts a scalar to all three components. The element type is
promoted from the arguments, so `Vec3(1, 2.0, 3)` is a `Vec3{Float64}` and a
`Vec3` holding `ForwardDiff.Dual`s works the same way as one holding floats.
"""
struct Vec3{T<:Real}
    x::T
    y::T
    z::T
end

Vec3(x::Real, y::Real, z::Real) = Vec3(promote(x, y, z)...)
Vec3(v::Real) = Vec3(v, v, v)
Vec3{T}(v::Real) where {T<:Real} = Vec3{T}(T(v), T(v), T(v))
Vec3{T}(v::Vec3) where {T<:Real} = Vec3{T}(T(v.x), T(v.y), T(v.z))

Base.convert(::Type{Vec3{T}}, v::Vec3) where {T<:Real} = Vec3{T}(v)
Base.eltype(::Type{Vec3{T}}) where {T} = T
Base.zero(::Type{Vec3{T}}) where {T} = Vec3{T}(zero(T), zero(T), zero(T))
Base.zero(v::Vec3) = zero(typeof(v))
Base.broadcastable(v::Vec3) = Ref(v)
Base.isapprox(a::Vec3, b::Vec3; kwargs...) =
    isapprox([a.x, a.y, a.z], [b.x, b.y, b.z]; kwargs...)
Base.show(io::IO, v::Vec3) = print(io, "Vec3(", v.x, ", ", v.y, ", ", v.z, ")")

Base.:+(a::Vec3, b::Vec3) = Vec3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::Vec3, b::Vec3) = Vec3(a.x - b.x, a.y - b.y, a.z - b.z)
Base.:-(a::Vec3) = Vec3(-a.x, -a.y, -a.z)
Base.:*(a::Vec3, b::Vec3) = Vec3(a.x * b.x, a.y * b.y, a.z * b.z)   # component-wise (colors)
Base.:*(s::Real, a::Vec3) = Vec3(s * a.x, s * a.y, s * a.z)
Base.:*(a::Vec3, s::Real) = s * a
Base.:/(a::Vec3, s::Real) = Vec3(a.x / s, a.y / s, a.z / s)

LinearAlgebra.dot(a::Vec3, b::Vec3) = a.x * b.x + a.y * b.y + a.z * b.z
LinearAlgebra.cross(a::Vec3, b::Vec3) =
    Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
LinearAlgebra.norm(a::Vec3) = sqrt(dot(a, a))
LinearAlgebra.normalize(a::Vec3) = a / norm(a)

_vec(v::Vec3) = v
_vec(v::Real) = Vec3(v)
