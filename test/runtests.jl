using Horizon
using Horizon: Ray, generate_ray, hit_distance
using LinearAlgebra: cross, dot, norm, normalize
using ForwardDiff
using Zygote
using Test

# Odd resolution so the central pixel's ray points exactly at the target.
const CAM = PinholeCamera(Vec3(0, 0, 5), Vec3(0, 0, 0); width = 33, height = 33, fov = 40)
const C = 17   # central row / column

# θ = albedo (3), light position (3), light intensity, shininess, sphere centre z
function build(θ)
    mat = Material(albedo = Vec3(θ[1], θ[2], θ[3]), specular = 0.25, shininess = θ[8])
    return Scene((Sphere(Vec3(0, 0, θ[9]), 1, mat),),
                 (PointLight(Vec3(θ[4], θ[5], θ[6]), θ[7]),); ambient = 0.02)
end
const θ0 = [0.7, 0.4, 0.3, 2.0, 3.0, 4.0, 50.0, 30.0, 0.1]

@testset "Horizon.jl" begin

@testset "Vec3" begin
    a, b = Vec3(1, 2, 3), Vec3(4.0, 5, 6)
    @test a isa Vec3{Int} && b isa Vec3{Float64}
    @test a + b == Vec3(5.0, 7, 9)
    @test a * b == Vec3(4.0, 10, 18)
    @test dot(a, b) == 32
    @test cross(Vec3(1, 0, 0), Vec3(0, 1, 0)) == Vec3(0, 0, 1)
    @test norm(normalize(b)) ≈ 1
    @test Vec3(0.5) == Vec3(0.5, 0.5, 0.5)
end

@testset "intersection" begin
    s = Sphere(Vec3(0, 0, 0), 1)
    r = Ray(Vec3(0, 0, 5), Vec3(0, 0, -1))
    @test hit_distance(s, r, 1e-4) ≈ 4
    @test isinf(hit_distance(s, Ray(Vec3(0, 3, 5), Vec3(0, 0, -1)), 1e-4))
    @test hit_distance(s, Ray(Vec3(0, 0, 0), Vec3(0, 0, -1)), 1e-4) ≈ 1   # from inside
    p = Plane(Vec3(0, -1, 0), Vec3(0, 2, 0))
    @test p.normal ≈ Vec3(0, 1, 0)
    @test hit_distance(p, Ray(Vec3(0, 1, 0), Vec3(0, -1, 0)), 1e-4) ≈ 2
    @test isinf(hit_distance(p, Ray(Vec3(0, 1, 0), Vec3(1, 0, 0)), 1e-4))
end

@testset "camera" begin
    r = generate_ray(CAM, C, C, 0.5, 0.5)
    @test r.d ≈ Vec3(0, 0, -1)
    @test generate_ray(CAM, 1, C, 0.5, 0.5).d.y > 0          # row 1 is the top
    @test generate_ray(CAM, C, 1, 0.5, 0.5).d.x < 0          # column 1 is the left
end

@testset "analytic radiance" begin
    # Lambertian sphere lit head-on by a directional light of irradiance π:
    # L = albedo/π · π · cos(0) = albedo at the centre pixel.
    albedo = Vec3(0.2, 0.5, 0.8)
    scene = Scene((Sphere(Vec3(0, 0, 0), 1, Material(; albedo)),),
                  (DirectionalLight(Vec3(0, 0, -1), π),); background = Vec3(0.1, 0.2, 0.3))
    img = render(scene, CAM)
    @test size(img) == (33, 33)
    @test img[C, C] ≈ albedo
    @test img[1, 1] == Vec3(0.1, 0.2, 0.3)             # corner sees background

    # Point light on the camera axis: irradiance falls off as 1/d².
    scene_p = Scene((Sphere(Vec3(0, 0, 0), 1, Material(; albedo)),),
                    (PointLight(Vec3(0, 0, 3), 4π),))
    @test render(scene_p, CAM)[C, C] ≈ albedo         # 4π / 2² · (1/π) = 1
end

@testset "shadows and ambient" begin
    m = Material(albedo = 0.5)
    scene = Scene((Sphere(Vec3(0, 0, 0), 1), Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), m)),
                  (DirectionalLight(Vec3(0, -1, 0), π),); ambient = 0.2)
    o = Vec3(6, -0.6, 0)                               # grazes past the sphere ...
    r = Ray(o, normalize(Vec3(0.5, -1, 0) - o))        # ... to a plane point under it
    @test radiance(scene, r) ≈ Vec3(0.2 * 0.5)                         # ambient only
    @test radiance(scene, r; shadows = false) ≈ Vec3(0.2 * 0.5 + 0.5)  # + albedo/π · π
end

@testset "number types" begin
    s32 = Scene((Sphere(Vec3(0f0, 0f0, 0f0), 1f0, Material(albedo = 0.5f0, shininess = 10f0)),),
                (PointLight(Vec3(2f0, 2f0, 4f0), 30f0),))
    cam32 = PinholeCamera(Vec3(0f0, 0f0, 5f0), Vec3(0f0, 0f0, 0f0); width = 9, height = 7)
    @test eltype(render(s32, cam32)) == Vec3{Float32}
    # A single Float64 anywhere promotes the whole render.
    @test eltype(render(s32, CAM)) == Vec3{Float64}
    @test size(render(s32, CAM; supersample = 2)) == (33, 33)
    # Supersampling a constant-radiance region leaves it unchanged.
    flat = Scene((Plane(Vec3(0, 0, -1), Vec3(0, 0, 1), Material(albedo = 0.5)),),
                 (DirectionalLight(Vec3(0, 0, -1), π),))
    @test render(flat, CAM; supersample = 3)[C, C] ≈ Vec3(0.5)
    @test size(to_array(render(s32, cam32))) == (7, 9, 3)
end

@testset "gradients vs finite differences" begin
    # Loss over pixels well inside the silhouette, so it is smooth in all
    # parameters (including the sphere position).
    window = (C-4):(C+4)
    loss(θ) = sum(v -> v.x + 2v.y + 3v.z, render(build(θ), CAM)[window, window])
    g = ForwardDiff.gradient(loss, θ0)
    h = 1e-6
    fd = map(eachindex(θ0)) do k
        e = zeros(length(θ0)); e[k] = h
        (loss(θ0 .+ e) - loss(θ0 .- e)) / 2h
    end
    @test all(!iszero, g)
    @test g ≈ fd rtol = 1e-5
end

@testset "forward and reverse mode agree" begin
    target = render(build([0.2, 0.6, 0.7, -2.0, 4.0, 3.0, 70.0, 60.0, 0.0]), CAM; supersample = 2)
    loss(θ) = mse(render(build(θ), CAM; supersample = 2), target)
    gf = ForwardDiff.gradient(loss, θ0)
    gz = only(Zygote.gradient(loss, θ0))
    @test gf ≈ gz rtol = 1e-10
end

@testset "inverse rendering recovers albedo and light" begin
    # Recover albedo and light position from a target image with plain Adam.
    fixed = θ0[7:9]
    θtrue = [0.2, 0.6, 0.7, -1.5, 3.5, 3.5]
    target = render(build(vcat(θtrue, fixed)), CAM)
    loss(p) = mse(render(build(vcat(p, fixed)), CAM), target)
    p = copy(θ0[1:6])
    m, v = zero(p), zero(p)
    lr, β1, β2 = [0.02, 0.02, 0.02, 0.1, 0.1, 0.1], 0.9, 0.999
    L0 = loss(p)
    for t in 1:300
        g = ForwardDiff.gradient(loss, p)
        m = β1 .* m .+ (1 - β1) .* g
        v = β2 .* v .+ (1 - β2) .* g .^ 2
        p = p .- lr .* (m ./ (1 - β1^t)) ./ (sqrt.(v ./ (1 - β2^t)) .+ 1e-8)
    end
    @test loss(p) < 1e-3 * L0
    @test p[1:3] ≈ θtrue[1:3] atol = 0.02
end

end
