"""
    PointLight(position, intensity)

An isotropic point light. `intensity` is radiant intensity (RGB or scalar);
irradiance falls off with the inverse square of distance.
"""
struct PointLight{T<:Real}
    position::Vec3{T}
    intensity::Vec3{T}
end

function PointLight(position::Vec3, intensity)
    I = _vec(intensity)
    T = float(promote_type(eltype(position), eltype(I)))
    return PointLight{T}(Vec3{T}(position), Vec3{T}(I))
end

"""
    DirectionalLight(direction, irradiance)

A distant light. `direction` is the direction the light *travels* (e.g.
`Vec3(0, -1, 0)` shines straight down); `irradiance` is RGB or scalar.
"""
struct DirectionalLight{T<:Real}
    direction::Vec3{T}
    irradiance::Vec3{T}
end

function DirectionalLight(direction::Vec3, irradiance)
    d, E = normalize(direction), _vec(irradiance)
    T = float(promote_type(eltype(d), eltype(E)))
    return DirectionalLight{T}(Vec3{T}(d), Vec3{T}(E))
end

numtype(::PointLight{T}) where {T} = T
numtype(::DirectionalLight{T}) where {T} = T

_convert(::Type{T}, l::PointLight) where {T} =
    PointLight{T}(Vec3{T}(l.position), Vec3{T}(l.intensity))
_convert(::Type{T}, l::DirectionalLight) where {T} =
    DirectionalLight{T}(Vec3{T}(l.direction), Vec3{T}(l.irradiance))

"""
    illuminate(light, x) -> (wi, Li, dist)

Unit direction from `x` towards the light, incident irradiance (at normal
incidence) and distance to the light.
"""
function illuminate(l::PointLight, x::Vec3)
    v = l.position - x
    d2 = dot(v, v)
    dist = sqrt(d2)
    return v / dist, l.intensity / d2, dist
end

illuminate(l::DirectionalLight{T}, ::Vec3) where {T} = (-l.direction, l.irradiance, T(Inf))
