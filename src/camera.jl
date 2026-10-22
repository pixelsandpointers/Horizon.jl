"""
    Ray(origin, direction)

A ray with origin `o` and (unit) direction `d`.
"""
struct Ray{T<:Real}
    o::Vec3{T}
    d::Vec3{T}
end

function Ray(o::Vec3, d::Vec3)
    T = promote_type(eltype(o), eltype(d))
    return Ray{T}(Vec3{T}(o), Vec3{T}(d))
end

(r::Ray)(t::Real) = r.o + t * r.d

"""
    PinholeCamera(eye, target; up = Vec3(0, 1, 0), fov = 40, width = 128, height = 96)

A pinhole camera at `eye` looking at `target`, with a vertical field of view
of `fov` degrees and an image of `width × height` pixels. `eye`, `target`,
`up` and `fov` are differentiable.
"""
struct PinholeCamera{T<:Real}
    eye::Vec3{T}
    forward::Vec3{T}
    right::Vec3{T}
    up::Vec3{T}
    tan_half_fov::T
    width::Int
    height::Int
end

function PinholeCamera(eye::Vec3, target::Vec3; up::Vec3 = Vec3(0, 1, 0),
                       fov::Real = 40, width::Integer = 128, height::Integer = 96)
    forward = normalize(target - eye)
    right = normalize(cross(forward, up))
    true_up = cross(right, forward)
    F = float(promote_type(eltype(eye), eltype(target), typeof(fov)))
    t = tan(deg2rad(F(fov)) / 2)
    T = float(promote_type(eltype(eye), eltype(forward), eltype(right), typeof(t)))
    return PinholeCamera{T}(Vec3{T}(eye), Vec3{T}(forward), Vec3{T}(right),
                            Vec3{T}(true_up), T(t), Int(width), Int(height))
end

numtype(::PinholeCamera{T}) where {T} = T

_convert(::Type{T}, c::PinholeCamera) where {T} =
    PinholeCamera{T}(Vec3{T}(c.eye), Vec3{T}(c.forward), Vec3{T}(c.right),
                     Vec3{T}(c.up), T(c.tan_half_fov), c.width, c.height)

"""
    generate_ray(camera, i, j, u, v)

Primary ray through pixel row `i`, column `j` (1-based, row 1 at the top),
at sub-pixel offset `(u, v) ∈ [0, 1)²`.
"""
function generate_ray(c::PinholeCamera{T}, i::Integer, j::Integer, u::Real, v::Real) where {T}
    aspect = T(c.width) / T(c.height)
    x = (2 * (j - 1 + u) / c.width - 1) * aspect * c.tan_half_fov
    y = (1 - 2 * (i - 1 + v) / c.height) * c.tan_half_fov
    return Ray(c.eye, normalize(c.forward + x * c.right + y * c.up))
end
