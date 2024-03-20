"""
Module for computing Dense Scale-Invariant Feature Transform (DSIFT) descriptors for object detection.

This module provides functions for computing DSIFT descriptors, which are widely used in computer vision tasks such as object detection. DSIFT descriptors are robust to changes in scale and rotation, making them suitable for detecting objects in images with varying viewpoints.

# Example
```julia
using DSIFTDescriptorForObjectDetection

# Load the query and test images
query_img = load("image1")
test_img = load("image2")
threshold = 0.6 # Example threshold, adjust based on your application

# Load image patches for the query & test image
query = DSIFTDescriptorForObjectDetection.loadImgByPatches(
    query_img,
    size(query_img),
    (8, 8)
)
test = DSIFTDescriptorForObjectDetection.loadImgByPatches(
    test_img,
    size(query_img),
    (8, 8)
)

# Compute descriptors for the query & test patches
query_descriptors = [DSIFTDescriptorForObjectDetection.DSIFT(query[1])]
test_descriptors = [DSIFTDescriptorForObjectDetection.DSIFT(test[i]) for i in 1:length(test)]

# Compute the average Euclidean distance between the query and test descriptors and find matches that exceed the threshold
dist = [DSIFTDescriptorForObjectDetection.computeAverageEuclidianDistance(query_descriptors[1], test_descriptors[i]) for i in 1:length(test_descriptors)]
results = [DSIFTDescriptorForObjectDetection.computeNegativeExponential(dist[i], alpha=2) for i in 1:length(dist)]
matches = DSIFTDescriptorForObjectDetection.findMatches(threshold, results)

# Generate the annoted image with bounding boxes
annotated_image = DSIFTDescriptorForObjectDetection.drawBoundingBoxes(test_img, matches, size(test_img), size(query_img), (8, 8))   
```
"""
module DSIFTDescriptorForObjectDetection

using FileIO, 
    Images,
    Colors,
    LinearAlgebra, 
    ImageView

include("./Maths.jl")
include("./Plots.jl")
include("./Utils.jl")

"""
    loadImgByPatches(image::AbstractMatrix, dim::Tuple{Int, Int}=(16, 16), stride::Tuple{Int, Int}=(8,8))

Load an image by dividing it into patches of specified dimensions and stride.

# Arguments
- `image`: The input image as an AbstractMatrix.
- `dim`: The dimensions of each patch as a tuple of integers. Default is (16, 16).
- `stride`: The stride between patches as a tuple of integers. Default is (8, 8).

# Returns
An array of patches extracted from the image.

# Example
```julia
image = rand(100, 100)
patches = loadImgByPatches(image, (16, 16), (8, 8))
```
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

"""
    DSIFT(patch::AbstractMatrix; sizePatch::Int=32)

Compute the Dense Scale-Invariant Feature Transform (DSIFT) descriptors for a given patch.

# Arguments
- `patch::AbstractMatrix`: The input patch for which to compute the descriptors.
- `sizePatch::Int`: The size of the subpatches used to compute the descriptors. Default is 32.

# Returns
An array of descriptors computed for the input patch.

# Example
```julia
patch = rand(32, 32)
descriptors = DSIFT(patch)
```
"""
function DSIFT(patch::AbstractMatrix; sizePatch::Int=32)
    # Convert patch to grayscale
    patch = Gray.(patch)
    
    blurred_patch = imfilter(patch, Kernel.gaussian(3))

    # Initialise an empty array to store the descriptors
    descriptors = []

    # Iterate over the patch in a sizePatchxsizePatch grid
    for i in 1:sizePatch:size(blurred_patch, 1)
        for j in 1:sizePatch:size(blurred_patch, 2)
            # Extract a sizePatchxsizePatch subpatch
            subpatch = blurred_patch[max(i, 1):min(i+(sizePatch-1), size(blurred_patch, 1)), max(j, 1):min(j+(sizePatch-1), size(blurred_patch, 2))]

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


end # module DSIFTDescriptorForObjectDetection

