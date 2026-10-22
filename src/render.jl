struct Hit{T<:Real}
    t::T
    n::Vec3{T}
    material::Material{T}
end

_nohit(::Type{T}) where {T} =
    Hit{T}(T(Inf), zero(Vec3{T}), Material{T}(zero(Vec3{T}), zero(Vec3{T}), one(T)))

# Ray-epsilon used for primary-ray tmin and shadow-ray offsets.
_eps(::Type{T}) where {T} = T(1e-4)

# Closest hit over a tuple of shapes, unrolled by recursion (AD-friendly, type-stable).
_closest(::Ray, ::Real, best::Hit) = best
function _closest(r::Ray, tmin::Real, best::Hit{T}, s, rest...) where {T}
    t = hit_distance(s, r, tmin)
    if t < best.t
        best = Hit{T}(t, surface_normal(s, r(t)), s.material)
    end
    return _closest(r, tmin, best, rest...)
end

_occluded(::Ray, ::Real, ::Real) = false
_occluded(r::Ray, tmin::Real, tmax::Real, s, rest...) =
    hit_distance(s, r, tmin) < tmax || _occluded(r, tmin, tmax, rest...)

# Direct lighting from each light, unrolled by recursion.
_direct(::Scene, x, n, wo, m, shadows, acc) = acc
function _direct(sc::Scene, x::Vec3{T}, n, wo, m, shadows, acc, light, rest...) where {T}
    wi, Li, dist = illuminate(light, x)
    cosθ = dot(n, wi)
    if cosθ > 0
        visible = !shadows ||
                  !_occluded(Ray(x + _eps(T) * n, wi), _eps(T), dist - _eps(T), sc.shapes...)
        if visible
            acc = acc + brdf(m, n, wi, wo) * Li * cosθ
        end
    end
    return _direct(sc, x, n, wo, m, shadows, acc, rest...)
end

"""
    radiance(scene, ray; shadows = true)

Outgoing radiance along `ray` (direct illumination plus ambient term).
Expects `scene` and `ray` to share one number type, as produced by [`render`](@ref).
"""
function radiance(sc::Scene, r::Ray{T}; shadows::Bool = true) where {T}
    hit = _closest(r, _eps(T), _nohit(T), sc.shapes...)
    isinf(hit.t) && return sc.background
    x = r(hit.t)
    wo = -r.d
    n = dot(hit.n, wo) < 0 ? -hit.n : hit.n      # shade the side facing the viewer
    m = hit.material
    return _direct(sc, x, n, wo, m, shadows, sc.ambient * m.albedo, sc.lights...)
end

function _pixel(sc::Scene, cam::PinholeCamera{T}, i::Int, j::Int, n::Int, shadows::Bool) where {T}
    if n == 1
        return radiance(sc, generate_ray(cam, i, j, T(0.5), T(0.5)); shadows)
    end
    samples = map(CartesianIndices((n, n))) do S
        radiance(sc, generate_ray(cam, i, j, (S[2] - T(0.5)) / n, (S[1] - T(0.5)) / n); shadows)
    end
    return sum(samples) / n^2
end

"""
    render(scene, camera; supersample = 1, shadows = true) -> Matrix{Vec3}

Render a `camera.height × camera.width` image of linear RGB radiance.

Each pixel is sampled on a deterministic `supersample × supersample` grid, so
the output is a smooth (noise-free) function of the scene parameters away from
visibility discontinuities. The number type of the image is the promotion of
every number in `scene` and `camera`, so dual numbers or `Float32`s propagate
through automatically.
"""
function render(scene::Scene, camera::PinholeCamera; supersample::Integer = 1,
                shadows::Bool = true)
    supersample ≥ 1 || throw(ArgumentError("supersample must be ≥ 1"))
    T = float(promote_type(numtype(scene), numtype(camera)))
    sc = _convert(T, scene)
    cam = _convert(T, camera)
    n = Int(supersample)
    return map(CartesianIndices((cam.height, cam.width))) do I
        _pixel(sc, cam, I[1], I[2], n, shadows)
    end
end
