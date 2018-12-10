#=
Learn a Sparse Basis for Natural Images
Author: Shashwat Shukla
Date: 9th March 2018
=#

using MAT # Import library to read .mat files
using Colors # Import library to convert array to image
using ImageView # Import library to display images
using Gtk.ShortNames # Import library to draw a canvas
using StatsBase # To compute mean and variance

# Define hyperparameters
const l = 12 # dimension of image patches
const d = l*l # dimension of flattened image patch
const n = 100 # number of basis vectors
const iter = 400000 # 400000 # number of iterations
const λ = 0.2 # Sparsity weight
const η = 0.001 # Learning rate
const batch_size = 1 # Batch size

file = matopen("IMAGES.mat") # Open file with whitened images
x = read(file, "IMAGES") # Extract the image matrix
close(file) # Close the file-stream

# Conjugate Descent for symmetric, positive semi-definite A
# Returns x that satisfies Ax=b
function conjugateDescent(A, θ, b)
    ϵ = 1e-5
    r = b - A*θ
    p = r
    while (norm(r) > ϵ)
        α = dot(r,r)/dot(p,A*p)
        θ = θ + α*p
        β = 1.0/dot(r,r)
        r = r - α*A*p
        β = β*dot(r,r)
        p = r + β*p
    end
    return θ
end

# Soft-thresholding function for use in LASSO regression
function soft_threshold(λ, β)
    if (β > λ)
        return (β - λ)
    elseif (β < -λ)
        return (β + λ)
    else
        return 0
    end
end

# Coordinate Descent solution to LASSO
function coordinateDescent(Φ,y,θ,λ)
    ϵ = 1e-5
    θ_old = θ + 1
    while (norm(θ-θ_old) > ϵ)
        θ_old = deepcopy(θ)
        for k = 1:n
            r = y - Φ*θ + Φ[:,k]*θ[k]
            θ[k] = soft_threshold(λ, dot(r,Φ[:,k])) / (norm(Φ[:,k])^2)
        end
    end
    return θ
end

# Automatic Relevance Determination
# This was the method finally used
function automaticRelevanceDetermination(Φ,y,θ)
    D = ones(n,1)
    W = zeros((n,n))
    for k = 1:10
        dθ = Φ'*(y-Φ*θ) - D.*θ
        ddθ = -Φ'*Φ - Diagonal(D[:,1])
        θ = θ - ddθ\dθ
        W = -inv(ddθ)
        D = 1./diag(W + θ*θ')
    end
    return θ, W
end

# Seed the random number generator
s = Dates.second(now())
srand(s)

# Initialise the basis vectors
Φ = rand(d,n) - 0.5
norm_Φ = sqrt.(sum(Φ.*Φ,1))
Φ = Φ ./ norm_Φ

# Iterate over training samples
for k = 1:iter
    ∇ = 0
    for g = 1:batch_size
        p = rand(1:501)
        q = rand(1:501)
        y = x[p:(p+l-1), q:(q+l-1), rand(1:10)] # Random 12x12 patch
        y = reshape(y,(d,1)) # Flatten y
        y = zscore(y) # Normalize y
        # E step
        θ = zeros(n,1)
        W = zeros((n,n))
        # A = Φ'*Φ + λ; b = Φ'*y # For Gaussian Prior
        # θ = conjugateDescent(A,θ,b) # For Gaussian Prior
        θ = coordinateDescent(Φ,y,θ,λ) # For Lasso
        # θ, W = automaticRelevanceDetermination(Φ,y,θ) # For ARD
        # M step
        ∇ = ∇ -y*θ' + Φ*(W + θ*θ')
    end
    # Update the basis vectors
    ∇ = ∇ / batch_size
    Φ = Φ - η*∇
    # Normalize the basis vectors
    norm_Φ = sqrt.(sum(Φ.*Φ,1))
    Φ = Φ ./ norm_Φ
end

# Φ = zscore(Φ)

file = matopen("basis.mat", "w")
write(file, "basis", Φ)
close(file)

# Display the learnt basis vectors
grid, frames, canvases = canvasgrid((10,10))
for k = 0:9
    for g = 1:10
        sample = Φ[:,10*k+g]
        sample = reshape(sample, (l,l))
        img = Gray.(sample)
        ImageView.imshow(canvases[k+1,g], img)
    end
end
win = Window(grid)
showall(win)

println("Done")
