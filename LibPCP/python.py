"""
Source: PCP Notebooks (https://www.audiolabs-erlangen.de/PCP)
Module: LibPCP.python
Author: Meinard Mueller, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
"""
import copy


def exercise_list(show_result=True):
    """Exercise 1: Basic List Manipulations
       Notebook: PCP_python.ipynb"""
    if show_result is False:
        return

    student_list = [[123, 'Meier', 'Sebastian'], [456, 'Smith', 'Walter']]
    print(student_list)

    student_list = student_list + [[789, 'Wang', 'Ming']]
    print(student_list)

    print(student_list[::-1])

    print(student_list[1][2])

    print(len(student_list))

    student_list_copy = copy.deepcopy(student_list)
    del(student_list_copy[0:2])
    student_list_copy[0][0] = 777
    print(student_list_copy)
    print(student_list)


def exercise_dict(show_result=True):
    """Exercise 2: Basic Dictionary Manipulations
       Notebook: PCP_python.ipynb"""
    if show_result is False:
        return

    student_dict = {123: ['Meier', 'Sebastian'], 456: ['Smith', 'Walter']}
    print(student_dict)

    student_dict[789] = ['Wang', 'Ming']
    print(student_dict)

    print(list(student_dict.keys()))

    print(list(student_dict.values()))

    print(student_dict[456][0])

    del student_dict[456]
    print(student_dict)

    print(len(student_dict))
