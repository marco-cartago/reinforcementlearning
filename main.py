import random
from enviroment import GridWorld
import time
import numpy as np

def main():
    np.random.seed(0)
    
    g = GridWorld(size=19)
    print(g)

    for i in range(10):
        
        a = g.get_actions()
        move = random.randint(0, len(a)-1)
        g.do_action(a, move)
        print(g, end="\r")
        time.sleep(0.05)
        if g.is_terminated:
            break
    
    print(g.get_episode())

if __name__ == "__main__":
    main()