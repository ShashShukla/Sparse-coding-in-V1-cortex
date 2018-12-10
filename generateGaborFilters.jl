#=
Generate Gabor Wavelets
Author: Shashwat Shukla
Date: 3rd April 2018
=#

using MAT # Import library to read .mat files
using Colors # Import library to convert array to image
using ImageView # Import library to display images
using Gtk.ShortNames # Import library to draw a canvas
using StatsBase # To compute mean and variance

# Define hyperparameters
const l = 12 # dimension of image patches
const d = l*l # dimension of flattened image patch
const n = 144 # number of basis vectors

# A general real 2D Gabor wavelet
function gabor(x,y,σ,a,b,α,ρ,ω,ψ)
    x̄ = (x-a)*cos(α) + (y-b)*sin(α)
    ȳ = (x-a)*sin(α) + (y-b)*cos(α)
    return 1/sqrt(2*pi*σ^2)*exp(-(x̄^2+(ρ*ȳ)^2)/(2*σ^2))*cos(2*pi*ω*x̄+ψ)
end

# Define range of various parameters
σ_ = 0.375^2 * [1,0.5,0.25,0.125, 0.0625]
ω_= [1,1.5,2.0,2.5,3.0,3.5,4.0,4.5]
ψ_ = [-pi/2,-pi/4,0,pi/4,pi/2]
ab_ = [[3,9],[3,3],[6,6],[9,9],[9,3]]
ρ = 0.60
α_ = pi * linspace(0,1,100)
# ab_ = [[3,9],[3,3],[6,6],[9,9],[9,3],[3,6],[9,6],[6,3],[6,9]]
# ω_= [0.3,0.5,0.7,0.9,1,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.5,5.0,5.5]

m = length(σ_) * length(ω_) * length(ψ_) * length(ab_) * length(ρ) * length(α_)
g = zeros(m,n) # The set Gabor filters
param = zeros(m,2) # Their corresponding spatial frequencies

k = 0
for σ in σ_, ω in ω_, ψ in ψ_, (a,b) in ab_, α in α_
    k = k + 1
    for i = 1:12, j = 1:12
        g[k,l*(i-1)+j] = gabor(i,j,σ,a,b,α,ρ,ω,ψ)
    end
    param[k,1] = ω * cos(α)
    param[k,2] = ω * sin(α)
end

# println(k) # Number of wavelets generated

# Normalize the filteres
norm_g = sqrt.(sum(g.*g,1))
g = g ./ norm_g

# Store the filters in a file
file = matopen("gabor.mat", "w")
write(file, "gabor", g)
write(file, "param", param)
close(file)

# z = reshape(g[1,:], (l,l))
# z = Gray.(z)
# imshow(z)

println("Done")
