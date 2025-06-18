[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-training/employee-scheduling?quickstart=1)
  
# The Employee Scheduling Problem With Symbolic Variables - Nonlinear (SOLUTION)

Exercise for D-Wave in-person training course to demonstrate the NL solver.

## Check the Original Program

Run ``nl_scheduling_preferences.py``. This program considers four employees
that need to be assigned to four open shifts.  Each employee has a
ranking/preference amongst the four shifts according to the image below,
where 1 is most preferred and 4 is least preferred.

![Employee preference rankings](scheduling_preferences.png "Employee Preferences")

Read through the code and take a look at how we're building up our
nonlinear model, or NL.  In particular, pay attention to:

1. Initialize the NL object with `model = Model()`.
2. Create labels for binary variables for each employee in each shift.
3. Create binary variable objects for each employee's shift.
4. Add constraints over employee binaries for each employee with `model.add_constraint(...)`.
5. Add objective terms as linear biases based on the employee preferences with `model.minimize(...)`.

## Exercise 1

For this exercise, we'll work with the file `nl_scheduling_addemployees.py`.
This file is very similar to `nl_scheduling_preferences.py`, and you will be
adding additional employees to the schedule.  Add the following employees
with their associated preferences for shifts 1-4 in the function `employee_preferences()`. 

1. Erik: [1,3,2,4]
2. Francis: [4,3,2,1]
3. Greta: [2,1,4,3]
4. Harry: [3,2,1,4]

When you run this problem, you should see two employees scheduled for each shift.

## Exercise 2

In this next exercise, we'll work with the file `nl_scheduling_restrictions.py`.
In this problem, we have 8 employees and 4 different shift options.
We've set up the initial NL model for all 8 employees with their preferences
over the 4 shifts. Now we need to take into account the following restrictions.

1. Anna is not able to work during shift 4.
2. Bill and Frank cannot work during the same shift.
3. Erica and Harriet would like to work the same shift.

Modify the function `build_nl()` to reflect these additional constraints.
Note that when you run your program, you may not have two employees per shift this time.

## Challenge: Exercise 3

For this final exercise, start with your completed file `nl_scheduling_restrictions.py`
from Exercise 2.  The optimal solution for Exercise 2 had some days with just 1 
person scheduled and others with many more.  Add a constraint to your NL so 
that each shift gets exactly two people scheduled.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
