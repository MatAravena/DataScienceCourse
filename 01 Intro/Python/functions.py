import random


import random
def get_error_term_coinflip():
    return random.choice([-3,+3])

get_error_term_coinflip()



def random_walk(start, steps, error_term_function, drift=0):
    """Performs a random_walk with a defined number of random_steps and returns the final position.
    
    Will call the error_term_function to determine the error_term and then consecutively call the
    random_step() function until the number ot steps is reached.
    
    Args:
        start (float): Starting Value in the random walk.
        steps (int): Number of steps taken in the walk (e.g. timespan you want to predict).
        error_term_function (function): Function Object to calculate the error. Must not accept Params.
        drift (float) : Drift term that will be applied to the random_step function. Defaults to 0.
 
    Returns:
        int: Final position of the random walk.
    """
    x_n = start
    
    for i in range(steps):
        error_term = error_term_function()
        x_n = random_step(x_n,error_term,drift)
   
    return x_n

random_walk(start=50, steps=30, error_term_function=get_error_term_coinflip)



random.seed(42)
print(random_walk(start=50, steps=30, error_term_function=get_error_term_coinflip))#14
print(random_walk(40,30,get_error_term_coinflip)) #46
print(random_walk(50,60,get_error_term_coinflip)) #50
print(random_walk(50,40,get_error_term_coinflip,5)) #214

def random_walk(start, steps, error_term_function, drift=0, full_detail=False):
    """Performs a random_walk with a defined number of random_steps and returns the final position.
    
    Will call the error_term_function to determine the error_term and then consecutively call the
    random_step() function until the number ot steps is reached.
    
    Args:
        start (float): Starting Value in the random walk.
        steps (int): Number of steps taken in the walk (e.g. timespan you want to predict).
        error_term_function (function): Function Object to calculate the error. Must not accept Params.
        drift (float) : Drift term that will be applied to the random_step function. Defaults to 0.
 
    Returns:
        int: Final position of the random walk.
    """
    x_n = start
    if full_detail:
        steps_collector = [x_n]
    
    for i in range(steps):
        error_term = error_term_function()
        x_n = random_step(x_n,error_term,drift)
        if full_detail:
            steps_collector.append(x_n)
    
    if full_detail:
        return steps_collector

random_walk(31, 30, get_error_term_coinflip)


import matplotlib.pyplot as plt

walk_path = random_walk(31,30,get_error_term_coinflip,full_detail=True)
plt.title("Books in BestBooks storage facility")
plt.ylabel("# Books in stock")
plt.xlabel("Days")
plt.plot(walk_path)
plt.show()




# Remember:

# The syntax for creating a function is:
# def my_function (my_params):
#   #execute code
#   return my_var
# You can set default values for function parameters with:
# def my_function(my_param=my_defaultvalue).