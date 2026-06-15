import numpy as np
import matplotlib.pyplot as plt
import random


lookup_table = {'111':7, # the keys are the neighborhoods patterns and the values the index of the rule to follow
                '110':6,
                '101':5,
                '100':4,
                '011':3,
                '010':2,
                '001':1,
                '000':0
                }


class OneDCellularAutomata:
    def __init__(self, starting_state, rule):
        self.starting_state = np.array(starting_state, dtype=int)
        assert len(self.starting_state.shape) < 2, 'The starting state should be 1D!'
        self.rule = format(rule, '08b')

    def update(self, n_steps):
        last_state = self.starting_state
        all_states = []
        for i in range(n_steps):
            all_states.append(last_state.copy())
            new_state = np.zeros(len(last_state), dtype=int)
            for j,n in enumerate(last_state):
                if j == 0:
                    neighborhood = f'{int(last_state[-1])}{int(last_state[0])}{int(last_state[1])}'
                elif j == len(last_state) - 1:
                    neighborhood = f'{int(last_state[-2])}{int(last_state[-1])}{int(last_state[0])}'
                else:
                    neighborhood = f'{int(last_state[j-1])}{int(last_state[j])}{int(last_state[j+1])}'
                
                cell_next_step_index = lookup_table[neighborhood]
                cell_next_step = str(self.rule)[cell_next_step_index]
                new_state[j] = int(cell_next_step)
            last_state = new_state

        evolution = np.array(all_states)
        plt.figure(figsize=(15,15))
        plt.imshow(evolution, aspect="auto", cmap='Greys')
        plt.colorbar()
        plt.show()



if __name__ == '__main__':
    starting_condition = [random.choice([0,1]) for i in range(600)]
    c = OneDCellularAutomata(starting_condition, rule=105)
    c.update(n_steps=600)