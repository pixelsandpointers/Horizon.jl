"""
    mse(a, b)

Mean squared error between two images (matrices of `Vec3`), averaged over
pixels and color channels. Differentiable.
"""
function mse(a::AbstractMatrix{<:Vec3}, b::AbstractMatrix{<:Vec3})
    size(a) == size(b) || throw(DimensionMismatch("images have sizes $(size(a)) and $(size(b))"))
    return sum(v -> dot(v, v), a .- b) / (3 * length(a))
end

"""
    to_array(img) -> Array{T,3}

Convert a `Matrix{Vec3{T}}` image to a `height × width × 3` array.
"""
to_array(img::AbstractMatrix{<:Vec3}) =
    [c == 1 ? img[i, j].x : c == 2 ? img[i, j].y : img[i, j].z
     for i in axes(img, 1), j in axes(img, 2), c in 1:3]

"""
    tonemap(c; exposure = 1, gamma = 2.2)

Scale linear radiance by `exposure`, clamp to [0, 1] and gamma-encode.
"""
function tonemap(c::Vec3; exposure::Real = 1, gamma::Real = 2.2)
    f(v) = clamp(float(v) * exposure, 0, 1)^(1 / gamma)
    return Vec3(f(c.x), f(c.y), f(c.z))
end

"""
    save_ppm(path, img; exposure = 1, gamma = 2.2)

Write an image to a binary PPM (P6) file after tone mapping. PPM needs no
image library; convert to PNG with any image tool.
"""
function save_ppm(path::AbstractString, img::AbstractMatrix{<:Vec3};
                  exposure::Real = 1, gamma::Real = 2.2)
    h, w = size(img)
    open(path, "w") do io
        write(io, "P6\n$w $h\n255\n")
        for i in 1:h, j in 1:w
            c = tonemap(img[i, j]; exposure, gamma)
            write(io, round(UInt8, 255 * c.x), round(UInt8, 255 * c.y), round(UInt8, 255 * c.z))
        end
    end
    return path
end
