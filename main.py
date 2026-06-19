import random
from enviroment import GridWorld
import time
import numpy as np

def main():
    np.random.seed(0)
    
    g = GridWorld(size=19)
    print(g)

    for i in range(100):
        
        a = g.get_actions()
        move = random.randint(0, len(a)-1)
        g.do_action(a, move)
        print(g, end="\r")

        if g.is_terminated:
            break

if __name__ == "__main__":
    main()