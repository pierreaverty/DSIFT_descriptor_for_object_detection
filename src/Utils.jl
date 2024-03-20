"""
    findMatches(threshold, results)

Find patches in the test image that exceed the given threshold.

# Arguments
- `threshold`: The threshold value to compare against.
- `results`: The array of similarity scores for each patch.

# Returns
An array of indices of patches that exceed the threshold.

# Example

```julia
threshold = 0.6
results = [0.5, 0.7, 0.8, 0.4, 0.9]
matches = findMatches(threshold, results)
```

"""
function findMatches(threshold, results)
    # Identify patches in the test image exceeding the threshold
    matches = findall(x -> x >= threshold, results)

    # Assuming you want to visualize or further analyze these matches
    for match in matches
        println("High similarity at patch index: ", match)
    end

    return matches
end