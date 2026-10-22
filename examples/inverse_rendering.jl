# Inverse rendering with Horizon: recover a sphere's albedo and a point light's
# position from a single target image, using forward-mode AD and Adam.
#
#   julia examples/inverse_rendering.jl
#
# Writes target.ppm, initial.ppm and optimized.ppm to the current directory.

# Set up the example environment (links the local Horizon checkout; works on Julia ≥ 1.10).
using Pkg
Pkg.activate(@__DIR__; io = devnull)
VERSION < v"1.11" && Pkg.develop(PackageSpec(path = dirname(@__DIR__)); io = devnull)  # no [sources] before 1.11
Pkg.instantiate()

using Horizon
using ForwardDiff
using Printf

const CAM = PinholeCamera(Vec3(0, 1.2, 6), Vec3(0, 0.2, 0); width = 160, height = 120, fov = 35)

# p = (albedo r, g, b, light x, y, z)
function scene(p)
    sphere = Sphere(Vec3(0, 0.5, 0), 1,
                    Material(albedo = Vec3(p[1], p[2], p[3]), specular = 0.3, shininess = 40))
    floor = Plane(Vec3(0, -0.5, 0), Vec3(0, 1, 0), Material(albedo = 0.6))
    light = PointLight(Vec3(p[4], p[5], p[6]), 120)
    return Scene((sphere, floor), (light,); ambient = 0.03)
end

render_p(p) = render(scene(p), CAM; supersample = 2)

p_true = [0.15, 0.45, 0.75, -2.5, 4.5, 3.0]
p_init = [0.80, 0.30, 0.20, 2.5, 3.0, 4.0]

target = render_p(p_true)
loss(p) = mse(render_p(p), target)

function adam(f, p; iters = 200, lr = [0.02, 0.02, 0.02, 0.1, 0.1, 0.1], β1 = 0.9, β2 = 0.999)
    m, v = zero(p), zero(p)
    for t in 1:iters
        g = ForwardDiff.gradient(f, p)
        m = β1 .* m .+ (1 - β1) .* g
        v = β2 .* v .+ (1 - β2) .* g .^ 2
        p = p .- lr .* (m ./ (1 - β1^t)) ./ (sqrt.(v ./ (1 - β2^t)) .+ 1e-8)
        (t == 1 || t % 25 == 0) && @printf("iter %4d   loss %.3e\n", t, f(p))
    end
    return p
end

p_opt = adam(loss, p_init)

@printf("\n%-14s %s\n", "", "albedo r/g/b           light x/y/z")
for (name, p) in (("initial", p_init), ("optimized", p_opt), ("ground truth", p_true))
    @printf("%-14s %s\n", name, join((@sprintf("%6.3f", x) for x in p), " "))
end

save_ppm("target.ppm", target)
save_ppm("initial.ppm", render_p(p_init))
save_ppm("optimized.ppm", render_p(p_opt))
