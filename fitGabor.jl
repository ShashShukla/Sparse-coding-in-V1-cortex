#=
Fitting Gabor wavelets to basis vectors learnt
through Sparse Coding
Author: Shashwat Shukla
Date: 3rd April 2018
=#

using MAT # Import library to read .mat files
using Colors # Import library to convert array to image
using ImageView # Import library to display images
using Gtk.ShortNames # Import library to draw a canvas
using StatsBase # To compute mean and variance
using PyCall # To invoke Python functions
pygui(:tk) # or pygui() ## this line is necessary
using PyPlot # To plot data

# Define hyperparameters
const l = 12 # dimension of image patches
const d = l*l # dimension of flattened image patch
const n = 100 # number of basis vectors

file = matopen("gabor.mat") # Open file with Gabor filters
g = read(file, "gabor") # Extract the set of generated Gabor filters
param = read(file, "param") # Extract corresponding parameters
close(file)

file = matopen("basis.mat") # Open file with whitened images
Φ = read(file, "basis") # Extract the image matrix
close(file)

# # Display the learnt basis vectors
# grid, frames, canvases = canvasgrid((10,10))
# for k = 0:9
#     for g = 1:10
#         sample = Φ[:,10*k+g]
#         sample = reshape(sample, (l,l))
#         img = Gray.(sample)
#         ImageView.imshow(canvases[k+1,g], img)
#     end
# end
# win = Window(grid)
# showall(win)

z = g*Φ # Compute the dot products
m = mapslices(indmax, z, 1)[:] # Find the maximum overlap
x = param[m, :] # Spatial frequencies for the best fits
scatter(x[:,1],x[:,2]) # Plot spatial frequencies of each wavelet
