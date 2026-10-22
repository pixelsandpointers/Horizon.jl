"""
    Horizon

A small, dependency-light differentiable renderer written in pure Julia.

Horizon is a direct-illumination ray tracer (spheres and planes, point and
directional lights, a Lambertian + normalized Blinn-Phong material). Every
type is parametric over its number type, so the whole renderer can be pushed
through any automatic-differentiation backend that works on plain Julia code
(ForwardDiff.jl, Zygote.jl, Enzyme.jl, ...). Horizon itself has no AD
dependency.

```julia
using Horizon
scene = Scene((Sphere(Vec3(0, 0, 0), 1, Material(albedo = Vec3(0.8, 0.3, 0.2))),),
              (PointLight(Vec3(3, 4, 5), 60),))
cam   = PinholeCamera(Vec3(0, 0, 5), Vec3(0, 0, 0); width = 128, height = 96)
img   = render(scene, cam)
```
"""
module Horizon

using LinearAlgebra: LinearAlgebra, dot, cross, norm, normalize

export Vec3
export Material
export Sphere, Plane
export PointLight, DirectionalLight
export Scene, PinholeCamera
export render, radiance
export mse, to_array, tonemap, save_ppm

include("vec3.jl")
include("camera.jl")
include("material.jl")
include("shapes.jl")
include("lights.jl")
include("scene.jl")
include("render.jl")
include("io.jl")

end # module Horizon
