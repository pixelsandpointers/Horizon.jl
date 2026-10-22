"""
    Material(; albedo = 0.8, specular = 0, shininess = 32)

A Lambertian diffuse lobe plus an energy-normalized Blinn-Phong specular lobe:

    f(wi, wo) = albedo / π + specular · (shininess + 8) / (8π) · max(n·h, 0)^shininess

`albedo` and `specular` are RGB reflectances (a scalar is broadcast to gray).
All three parameters are differentiable.
"""
struct Material{T<:Real}
    albedo::Vec3{T}
    specular::Vec3{T}
    shininess::T
end

function Material(; albedo = 0.8, specular = 0, shininess::Real = 32)
    a, s = _vec(albedo), _vec(specular)
    T = float(promote_type(eltype(a), eltype(s), typeof(shininess)))
    return Material{T}(Vec3{T}(a), Vec3{T}(s), T(shininess))
end

numtype(::Material{T}) where {T} = T

_convert(::Type{T}, m::Material) where {T} =
    Material{T}(Vec3{T}(m.albedo), Vec3{T}(m.specular), T(m.shininess))

"""
    brdf(material, n, wi, wo)

Evaluate the BRDF for unit normal `n`, light direction `wi` and view
direction `wo` (both pointing away from the surface).
"""
function brdf(m::Material, n::Vec3, wi::Vec3, wo::Vec3)
    diffuse = m.albedo / π
    h = normalize(wi + wo)
    c = dot(n, h)
    # Branch instead of max(c, 0)^s so that ∂/∂shininess never sees log(0).
    if c > 0
        return diffuse + m.specular * ((m.shininess + 8) / 8 / π * c^m.shininess)
    else
        return diffuse
    end
end
