###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name: Amaan Karim Ahmad


from ps1_partition import get_partitions
import time
import operator

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    print("Loading words from file...")
    # inFile: file
    inFile = open(filename, 'r')
    # wordlist: list of strings
    wordlist = {}
    for line in inFile:
        cow = line.split(',')
        wordlist[cow[0]] = int(cow[1]) # 0: name, 1: weight
    inFile.close()
    print("  ", len(wordlist), "words loaded.")
    return wordlist

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # Sort the dictionary to go in a descending order
    sorted_cows = sorted(cows, key = cows.get, reverse=True) # cows.get apply to all the keys in dictionary
    # Intialise list of lists for the cows transported
    trips = []
    # Whilst there are cows to take on the trip
    while len(sorted_cows) > 0:
        # Initialise the value for the weight of cows on a trip
        total_weight = 0
        # Intialise each individual trip in all the trips
        trip = []
        for name in sorted_cows[:]:
            # If cow can be taken on trip, add name to trip
            if(cows[name]+total_weight) <= limit:
                trip.append(name)
                # Calculate the new total_weight
                total_weight = total_weight + cows[name]
                # Remove the name 
                sorted_cows.remove(name)
        # Add the trip to trips list
        trips.append(trip)
    # Return list of lists
    return trips

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # Sort the dictionary to go in a descending order
    sorted_cows = sorted(cows, key = cows.get, reverse=True) # cows.get apply to all the keys in dictionary
    # Get all possible partitions of the cows
    cow_partitions = get_partitions(sorted_cows[:])
    # Intialise variables to track best trip possible
    best_trip_combo = sorted_cows[:]
    for partition in cow_partitions:
        success = 0
        # Check each trip in the partition is doable
        for trip in partition:
            trip_weight = 0
            for cow in trip:
                # Obtain the weight of the cow
                for cow_name in sorted_cows[:]:
                    if cow == cow_name:
                        trip_weight += cows[cow_name]
            if trip_weight > limit:
                success = 0
                break
            else:
                success += 1
        if success == len(partition):
            # Change the best trip if needed
            if len(partition) < len(best_trip_combo):
                best_trip_combo = []
                best_trip_combo.append(partition)
    return best_trip_combo
        
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    print()
    # Load the cows from the txt file
    cows = load_cows("ps1_cow_data.txt")
    print()
    print("---Greedy algorithm---")
    print()
    start = time.time()
    print(greedy_cow_transport(cows))
    end = time.time()
    print()
    print(end - start, "seconds")
    print()
    print("---Brute force algorithm---")
    print()
    start = time.time()
    print(brute_force_cow_transport(cows))
    end = time.time()
    print()
    print(end - start, "seconds")
    print()

compare_cow_transport_algorithms()
