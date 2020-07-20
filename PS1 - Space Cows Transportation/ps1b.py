###########################
# 6.0002 Problem Set 1b: Space Change
# Name: Amaan Ahmad

#================================
# Part B: Golden Eggs
#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    # Base case
    if target_weight == 0:
            return 0
    # Return already stored value
    try:
        return memo[target_weight]
    # If there isn't a stored value then calculate and store it
    except KeyError:
        # Iterating through the different egg weights
        for egg in egg_weights:
            # Subtracted from the target
            new_weight = target_weight - egg
            # Egg able to be taken on board
            if new_weight >= 0:
                # Add another egg to the total eggs taken
                eggs_taken = 1 + dp_make_weight(egg_weights, new_weight, memo)
                # Add dictionary entry
                memo[target_weight] = eggs_taken
    return eggs_taken


# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print("\n")
    
    egg_weights = (1, 5, 10, 25)
    n = 25 
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 25")
    print("Expected ouput: 1 (1 * 25 = 25)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print("\n")

    egg_weights = (1, 5, 10, 25)
    n = 4 
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 4")
    print("Expected ouput: 4 (4 * 1 = 1)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print("\n")