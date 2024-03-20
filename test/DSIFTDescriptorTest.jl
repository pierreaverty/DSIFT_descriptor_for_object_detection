# Include the DSIFTDescriptorForObjectDetection module
include("../src/DSIFTDescriptorForObjectDetection.jl")

# Import necessary packages
using FileIO,
    Images

# Load the query and test images
query_img = load("data/query2.png")
test_img = load("data/test.jpg")
threshold = 0.6 # Example threshold, adjust based on your application

# Resize the query image
query_img = imresize(query_img, ratio=1)

# Print the size of the query and test images
print(size(query_img))
print(size(test_img))

# Load image patches for the query image
query = DSIFTDescriptorForObjectDetection.loadImgByPatches(
    query_img,
    size(query_img),
    (8, 8)
)

# Load image patches for the test image
test = DSIFTDescriptorForObjectDetection.loadImgByPatches(
    test_img,
    size(query_img),
    (8, 8)
)

# Print the size of the query and test patches
println("query patches size: ", size(query))
println("test patches size: ", size(test))

# Compute descriptors for the query patches
query_descriptors = [DSIFTDescriptorForObjectDetection.DSIFT(query[1])]

# Compute descriptors for the test patches
test_descriptors = [DSIFTDescriptorForObjectDetection.DSIFT(test[i]) for i in 1:length(test)]

# Print the size of the query and test descriptors
println("query descriptors size: ", size(query_descriptors))
println("test descriptors size: ", size(test_descriptors))

dist = [DSIFTDescriptorForObjectDetection.computeAverageEuclidianDistance(query_descriptors[1], test_descriptors[i]) for i in 1:length(test_descriptors)]

results = [DSIFTDescriptorForObjectDetection.computeNegativeExponential(dist[i], alpha=2) for i in 1:length(dist)]

matches = DSIFTDescriptorForObjectDetection.findMatches(threshold, results)

# Visualize matches
annotated_image = DSIFTDescriptorForObjectDetection.drawBoundingBoxes(test_img, matches, size(test_img), size(query_img), (8, 8))

save("results/annotated_test_image.jpg", annotated_image)
