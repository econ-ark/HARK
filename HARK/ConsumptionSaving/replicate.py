'''
This module contains simple code to copy the current sub-module to a location on 
your hard drive, as specified by you. The purpose of this code is to provide the 
user with a simple way to access a *copy* of code that usually sits deep in the 
Econ-ARK package structure, for purposes of tinkering and experimenting directly 
yourself. This is meant to be a simple way to explore HARK code. To interact 
with the codebase under active development, please refer to the documentation 
under github.com/econ-ark/HARK/
'''
from __future__ import division, print_function
import os
from distutils.dir_util import copy_tree


#  Define a function to run the copying:
def copy_module(target_path, my_directory_full_path, my_module):

    if target_path == 'q' or target_path == 'Q':
        print("Goodbye!")
        return
    elif target_path == os.path.expanduser("~") or os.path.normpath(target_path) == os.path.expanduser("~"):
        print("You have indicated that the target location is "+target_path+" -- that is, you want to wipe out your home directory with the contents of "+my_module+". My programming does not allow me to do that.\n\nGoodbye!")
        return
    elif os.path.exists(target_path):
        print("There is already a file or directory at the location "+target_path+". For safety reasons this code does not overwrite existing files.\nPlease remove the file at "+target_path+" and try again.")
        return
    else:
        user_input = input("""You have indicated you want to copy module:\n    """+ my_module
                           + """\nto:\n    """+ target_path +"""\nIs that correct? Please indicate: y / [n]\n\n""")
        if user_input == 'y' or user_input == 'Y':
            #print("copy_tree(",my_directory_full_path,",", target_path,")")
            copy_tree(my_directory_full_path, target_path)
        else:
            print("Goodbye!")
            return
  
        


def replicate():
    """Find all the things needed to copy the file locations. In part this is to
    add a layer of protection against accidentally wiping out the user home 
    directory."""

    # Find a default directory -- user home directory:
    home_directory_RAW = os.path.expanduser("~")
        # Thanks to https://stackoverflow.com/a/4028943

    # Find the directory of the file being called:
    my_directory_full_path = os.path.dirname(os.path.realpath(__file__))
    # From https://stackoverflow.com/a/5137509
    # Important note from that answer: 
    #   (Note that the incantation above won't work if you've already used os.chdir() to change your current working directory, since the value of the __file__ constant is relative to the current working directory and is not changed by an os.chdir() call.)
    #
    # NOTE: for this specific file that I am testing, the path should be:
    # '/home/npalmer/anaconda3/envs/py3fresh/lib/python3.6/site-packages/HARK/SolvingMicroDSOPs/---example-file---

    head_path, my_module = os.path.split(my_directory_full_path)

    home_directory_with_module = os.path.join(home_directory_RAW, my_module)

    # Interact with the user:
    #     - Ask the user for the target place to copy the directory
    #         - Offer use "q/y/other" option
    #     - Check if there is something there already
    #     - If so, ask if should replace
    #     - If not, just copy there
    #     - Quit

    target_path = input("""You have invoked the 'replicate' process for the current module:\n    """ + 
                        my_module + """\nThe default copy location is your home directory:\n    """+
                        home_directory_with_module +"""\nPlease enter one of the three options in single quotes below, excluding the quotes: 
                        
        'q' or return/enter to quit the process
        'y' to accept the default home directory: """+home_directory_with_module+"""
        'n' to specify your own pathname\n\n""")

    
    if target_path == 'n' or target_path == 'N':
        target_path = input("""Please enter the full pathname to your target directory location: """)
        
        # Clean up:
        target_path = os.path.expanduser(target_path)
        target_path = os.path.expandvars(target_path)
        target_path = os.path.normpath(target_path)
        
        # Check to see if they included the module name; if not add it here:
        temp_head, temp_tail = os.path.split(target_path)
        if temp_tail != my_module:
            target_path = os.path.join(target_path, my_module)
        
    elif target_path == 'y' or target_path == 'Y':
        # Just using the default path:
        target_path = home_directory_with_module
    else:
        # Assume "quit"
        return 
    
    if target_path != 'q' and target_path != 'Q' or target_path == '':   
        # Run the copy command:
        copy_module(target_path, my_directory_full_path, my_module)   
    
    return

