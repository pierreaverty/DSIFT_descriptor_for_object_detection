module DSIFTDescriptorForObjectDetection

using FileIO, 
    Images,
    ImageFeatures

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

function loadImgByPatches(image::AbstractMatrix, dim::Tuple{Int, Int}=(16, 16), stride::Tuple{Int, Int}=(8,8))
    patches = []

    for i in 1:stride[1]:size(image, 1)-(dim[1]-1)
        for j in 1:stride[2]:size(image, 2)-(dim[2]-1)
            patch = image[i:i+(dim[1]-1), j:j+(dim[2]-1)]
            push!(patches, patch)
        end
    end
    return patches
end

"""
    computeSIFTDescriptors(patch::AbstractMatrix)

Compute SIFT descriptors for the given patch.

# Arguments
- `patch::AbstractMatrix`: The patch for which to compute SIFT descriptors.

# Returns
- `descriptors`: An empty array to store the computed SIFT descriptors.

"""
function computeSIFTDescriptors(patch::AbstractMatrix)
    sift = SIFT()
    descriptors = []
    sift_features = sift(patch)

    return descriptors
end



end # module DSIFTDescriptorForObjectDetection

using FileIO

img = load("data/query.jpg")

patches = DSIFTDescriptorForObjectDetection.loadImgByPatches(
    img,
    (16, 16),
    (8, 8)
)

DSIFTDescriptorForObjectDetection.computeSIFTDescriptors(patches[1])
