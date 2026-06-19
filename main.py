import random
from enviroment import GridWorld
import time

def main():
    g = GridWorld()
    print(g)
    for i in range(100000):
        a = g.get_actions()
        move = random.randint(0, len(a)-1)
        g.do_action(a[move])
        print(g, end="\r")
        time.sleep(0.01)
        if g.is_terminated:
            break

if __name__ == "__main__":
    main()