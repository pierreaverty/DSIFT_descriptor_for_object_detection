
"""
    drawBoundingBoxes(image, matches, imageSize, patchSize, stepSize)

Draws bounding boxes on an image based on the given matches.

# Arguments
- `image`: The input image.
- `matches`: A collection of matches.
- `imageSize`: The size of the image.
- `patchSize`: The size of the patches.
- `stepSize`: The step size for sliding the patches.

# Returns
The annotated image with bounding boxes.

"""

function drawBoundingBoxes(image, matches, imageSize, patchSize, stepSize)
    annotated_image = copy(image)
    
    for match in matches
        (x, y) = indexToCoordinates(match, imageSize, patchSize, stepSize)
        # Draw top and bottom borders
        for i in x:min(x+patchSize[2]-1, imageSize[2])
            annotated_image[y, i] = RGB(1,0,0) # Top border
            annotated_image[min(y+patchSize[1]-1, imageSize[1]), i] = RGB(1,0,0) # Bottom border
        end
        # Draw left and right borders
        for j in y:min(y+patchSize[1]-1, imageSize[1])
            annotated_image[j, x] = RGB(1,0,0) # Left border
            annotated_image[j, min(x+patchSize[2]-1, imageSize[2])] = RGB(1,0,0) # Right border
        end
    end
    
    return annotated_image
end

"""
    indexToCoordinates(index, imageSize, patchSize, stepSize)

Converts a linear index to the corresponding coordinates in a grid.

# Arguments
- `index`: The linear index.
- `imageSize`: A tuple representing the size of the image.
- `patchSize`: A tuple representing the size of the patch.
- `stepSize`: A tuple representing the step size.

# Returns
A tuple `(x, y)` representing the coordinates of the top-left corner of the patch.

"""

function indexToCoordinates(index, imageSize, patchSize, stepSize)
    rows = 1:stepSize[1]:(imageSize[1]-patchSize[1]+1)
    cols = 1:stepSize[2]:(imageSize[2]-patchSize[2]+1)

    # Calculate row and column in the grid
    row = ceil(Int, index / length(cols))
    col = index % length(cols)
    col = col == 0 ? length(cols) : col  # Correct for modulo behavior

    # Calculate the top-left corner of the patch
    y = rows[row]
    x = cols[col]

    return (x, y)
end