# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# imports
from dwave.optimization import Model
from dwave.system import LeapHybridNLSampler

import numpy as np

# Set the solver we're going to use
def set_sampler():
    '''Returns an optimization sampler'''

    sampler = LeapHybridNLSampler()

    return sampler

# Set employees and preferences
def employee_preferences():
    '''Returns a dictionary of employees with their preferences'''

    preferences = { "Anna": [1,2,3,4],
                    "Bill": [3,2,1,4],
                    "Chris": [4,2,3,1],
                    "Diane": [4,1,2,3],
                    "Erica": [1,2,3,4],
                    "Frank": [3,2,1,4],
                    "George": [4,2,3,1],
                    "Harriet": [4,1,2,3]}

    return preferences

# Create NL object
def build_nl():
    '''Builds the NL for our problem'''
    model = Model()

    preferences = employee_preferences()
    shift_pref = model.constant(np.array(list(preferences.values())))
    num_shifts = 4
    names = list(preferences.keys())
    num_employee = len(names)
    variables = model.binary([num_employee, num_shifts])

    model.minimize((shift_pref * variables).sum())
    for i in range(num_employee):
        model.add_constraint(variables[i].sum() == model.constant(1.0))

    # TODO: Restrict Anna from working shift 4
    #model.add_constraint()

    # TODO: Set constraints to reflect the restrictions in the README.
    # Bill and Frank CANNOT woek together
    for shift in range(num_shifts):
        model.add_constraint((variables[names.index("Bill")][shift] + variables[names.index("Frank")][shift]) <= model.constant(1.0))

    # Requires that Erica and harriet work together
    model.add_constraint(sum((variables[names.index("Erica")][shift] * variables[names.index("Harriet")][shift] for shift in range(num_shifts)), start=model.constant(0.0)) >= model.constant(1.0))

    # All shifts to have 2 employees
    for shift in range(num_shifts):
        model.add_constraint(sum((variables[i][shift] for i in range(num_employee)), start=model.constant(0.0)) == model.constant(2.0))

    return model

# Solve the problem
def solve_problem(model, sampler):
    '''Runs the provided model object on the designated sampler'''

    # Initialize the NL solver
    #sampler = set_sampler()

    # Solve the problem using the NL solver
    sampleset = sampler.sample(model, label='Training - Employee Scheduling')
    
    return sampleset.result()

# Process solution
def process_sampleset(model, sampleset):
    '''Processes the best solution found for displaying'''
   
    # Get the first solution
    decision_vars = list(model.iter_decisions())[0]

    shift_schedule = [[] for _ in range(4)]
    preferences = employee_preferences()
    names = list(preferences.keys())
    num_employees = len(names)

    indexed_vars = [[decision_vars[i][j] for j in range(num_shifts)] for i in range(num_employees)]

    with model.lock():
        for i in range(num_employees):
            for j in range(num_shifts):
                val = indexed_vars[i][j].state(0)
                if val == 1.0:
                    shift_schedule[j].append(names[i])
    
    return shift_schedule

## ------- Main program -------
if __name__ == "__main__":

    # Problem information
    shifts = [1, 2, 3, 4]
    num_shifts = len(shifts)

    model = build_nl()

    sampler = set_sampler()

    sampleset = solve_problem(model, sampler)

    shift_schedule = process_sampleset(model, sampleset)

    for i in range(num_shifts):
        print("Shift:", shifts[i], "\tEmployee(s): ", shift_schedule[i])
