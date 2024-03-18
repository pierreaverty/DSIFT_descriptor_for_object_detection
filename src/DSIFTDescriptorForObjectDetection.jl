module DSIFTDescriptorForObjectDetection

using FileIO, 
    Images,
    Colors

"""
    loadImgByPatches(image::AbstractMatrix, dim::Tuple{Int, Int}=(16, 16), stride::Tuple{Int, Int}=(8,8))

Load an image by dividing it into patches of specified dimensions and stride.

# Arguments
- `image`: The input image as an AbstractMatrix.
- `dim`: The dimensions of each patch as a tuple of integers. Default is (16, 16).
- `stride`: The stride between patches as a tuple of integers. Default is (8, 8).

# Returns
An array of patches extracted from the image.

"""

function loadImgByPatches(image::AbstractMatrix, patchSize::Tuple{Int, Int}=(16, 16), stepSize::Tuple{Int, Int}=(8,8))
    patches = []

    for i in 1:stepSize[1]:size(image, 1)-(patchSize[1]-1)
        for j in 1:stepSize[2]:size(image, 2)-(patchSize[2]-1)
            patch = image[i:i+(patchSize[1]-1), j:j+(patchSize[2]-1)]
            push!(patches, patch)
        end
    end
    return patches
end

function DSIFT(patch::AbstractMatrix)
    # Convert patch to grayscale
    patch = Gray.(patch)
    
    blurred_patch = imfilter(patch, Kernel.gaussian(3))

    # Initialize an empty array to store the descriptors
    descriptors = []

    # Iterate over the patch in a 4x4 grid
    for i in 1:4:size(blurred_patch, 1)
        for j in 1:4:size(blurred_patch, 2)
            # Extract a 4x4 subpatch
            subpatch = blurred_patch[i:i+3, j:j+3]
            
            # Compute the gradients of the subpatch
            magnitude, orientation = computeGradients(subpatch)

            # Create a histogram of the gradients  
            descriptor = createHistogram(magnitude, orientation)
            
            # Append the descriptor to the array
            push!(descriptors, descriptor)
        end
    end

    # Return the array of descriptors
    return descriptors
end

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

function createHistogram(magni::AbstractMatrix, orientation::AbstractMatrix, bins::Int=8)
    # Initialize an empty array to store the histogram
    histogram = zeros(bins)
    
    # Iterate over the pixels in the patch
    for i in 1:size(magni, 1)
        for j in 1:size(magni, 2)
            # Compute the bin index
            bin = Int(floor((orientation[i, j] + π) / (2π) * bins) + 1)
            
            # Increment the corresponding bin in the histogram
            histogram[bin] += magni[i, j]
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

end # module DSIFTDescriptorForObjectDetection
