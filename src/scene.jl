"""
    Scene(shapes, lights; ambient = 0, background = 0)

A collection of shapes and lights. Pass `shapes` and `lights` as tuples
(vectors are converted) so heterogeneous scenes stay type-stable.
`ambient` is a constant radiance multiplied by each surface's albedo;
`background` is the radiance of rays that escape the scene.
"""
struct Scene{T<:Real, S<:Tuple, L<:Tuple}
    shapes::S
    lights::L
    ambient::Vec3{T}
    background::Vec3{T}
end

function Scene(shapes, lights; ambient = 0, background = 0)
    s, l = Tuple(shapes), Tuple(lights)
    a, b = _vec(ambient), _vec(background)
    T = promote_type(eltype(a), eltype(b))   # not floated: Int defaults must not force Float64
    return Scene{T, typeof(s), typeof(l)}(s, l, Vec3{T}(a), Vec3{T}(b))
end

_numtype(::Tuple{}) = Bool
_numtype(t::Tuple) = promote_type(map(numtype, t)...)

numtype(sc::Scene{T}) where {T} = promote_type(T, _numtype(sc.shapes), _numtype(sc.lights))

function _convert(::Type{T}, sc::Scene) where {T}
    s = map(x -> _convert(T, x), sc.shapes)
    l = map(x -> _convert(T, x), sc.lights)
    return Scene{T, typeof(s), typeof(l)}(s, l, Vec3{T}(sc.ambient), Vec3{T}(sc.background))
end
