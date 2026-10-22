"""
    Sphere(center, radius, material = Material())
"""
struct Sphere{T<:Real}
    center::Vec3{T}
    radius::T
    material::Material{T}
end

function Sphere(center::Vec3, radius::Real, material::Material = Material())
    T = float(promote_type(eltype(center), typeof(radius), numtype(material)))
    return Sphere{T}(Vec3{T}(center), T(radius), _convert(T, material))
end

"""
    Plane(point, normal, material = Material())

An infinite plane through `point` with normal `normal` (normalized on construction).
"""
struct Plane{T<:Real}
    point::Vec3{T}
    normal::Vec3{T}
    material::Material{T}
end

function Plane(point::Vec3, normal::Vec3, material::Material = Material())
    n = normalize(normal)
    T = float(promote_type(eltype(point), eltype(n), numtype(material)))
    return Plane{T}(Vec3{T}(point), Vec3{T}(n), _convert(T, material))
end

numtype(::Sphere{T}) where {T} = T
numtype(::Plane{T}) where {T} = T

_convert(::Type{T}, s::Sphere) where {T} =
    Sphere{T}(Vec3{T}(s.center), T(s.radius), _convert(T, s.material))
_convert(::Type{T}, p::Plane) where {T} =
    Plane{T}(Vec3{T}(p.point), Vec3{T}(p.normal), _convert(T, p.material))

"""
    hit_distance(shape, ray, tmin)

Distance along `ray` to the nearest intersection with `shape` beyond `tmin`,
or `Inf` if there is none. Assumes `ray.d` is unit length.
"""
function hit_distance(s::Sphere, r::Ray, tmin::Real)
    oc = r.o - s.center
    b = dot(oc, r.d)
    c = dot(oc, oc) - s.radius^2
    disc = b^2 - c
    T = typeof(disc)
    disc < 0 && return T(Inf)
    sq = sqrt(disc)
    t1 = -b - sq
    t1 > tmin && return t1
    t2 = -b + sq
    return t2 > tmin ? t2 : T(Inf)
end

function hit_distance(p::Plane, r::Ray, tmin::Real)
    denom = dot(p.normal, r.d)
    T = typeof(denom)
    abs(denom) < sqrt(eps(T)) && return T(Inf)
    t = dot(p.point - r.o, p.normal) / denom
    return t > tmin ? t : T(Inf)
end

surface_normal(s::Sphere, x::Vec3) = (x - s.center) / s.radius
surface_normal(p::Plane, ::Vec3) = p.normal
