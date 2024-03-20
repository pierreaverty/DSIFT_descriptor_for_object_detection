"""
    createHistogram(magni::AbstractMatrix, orientation::AbstractMatrix, bins::Int=8)

Create a histogram of orientations for a given patch.

# Arguments
- `magni::AbstractMatrix`: A matrix representing the magnitudes of the patch.
- `orientation::AbstractMatrix`: A matrix representing the orientations of the patch.
- `bins::Int=8`: The number of bins in the histogram. Default is 8.

# Returns
- `histogram`: An array representing the histogram of orientations.
"""

function createHistogram(magnitude::AbstractMatrix, orientation::AbstractMatrix, bins::Int=8)
    # Initialize an empty array to store the histogram
    histogram = zeros(bins)
    
    # Iterate over the pixels in the patch
    for i in 1:size(magnitude, 1)
        for j in 1:size(magnitude, 2)
            # Compute the bin index
            bin = mod(Int(floor((orientation[i, j] + π) / (2π) * bins)), bins) + 1

            # Increment the corresponding bin in the histogram
            histogram[bin] += magnitude[i, j]
        end
    end

    # Return the histogram
    return histogram
end

"""
    computeGradients(patch::AbstractMatrix)

Compute the gradients in the x and y directions of the given patch.

# Arguments
- `patch::AbstractMatrix`: The patch for which to compute the gradients.

# Returns
- `magnitude`: The magnitude of the gradients.
- `orientation`: The orientation of the gradients.
"""
function computeGradients(patch::AbstractMatrix)
    # Define the x and y sobel kernels
    x, y  = Kernel.sobel()
    # Compute the gradients in the x and y directions
    gx = imfilter(patch, x)
    gy = imfilter(patch, y)
    
    # Compute the magnitude and orientation of the gradients
    magnitude = sqrt.(gx.^2 .+ gy.^2)
    orientation = atan.(gy, gx)
    
    return magnitude, orientation
end

"""
    computeAverageEuclidianDistance(descriptor1::AbstractArray, descriptor2::AbstractArray)

Compute the average Euclidean distance between two descriptors.

# Arguments
- `descriptor1::AbstractArray`: The first descriptor.
- `descriptor2::AbstractArray`: The second descriptor.

# Returns
The average Euclidean distance between the two descriptors.
"""
function computeAverageEuclidianDistance(descriptor1::AbstractArray, descriptor2::AbstractArray)
    # Compute the sum of the squared differences between the two descriptors
    res = 0
    for i in 1:length(descriptor1)
        res += sum((descriptor1[i] .- descriptor2[i]).^2)
    end
    
    # Return the square root of the sum
    return 1/length(descriptor1) .* sqrt(res)
end

"""
    computeNegativeExponential(dist::Float64, alpha::Int=2)

Compute the negative exponential of the given distance.

# Arguments
- `dist::Float64`: The distance value.
- `alpha::Int`: The exponent value. Default is 2.

# Returns
The result of the negative exponential computation.

"""
function computeNegativeExponential(dist::Float64; alpha::Int=2)
    return alpha^(-dist)
end
