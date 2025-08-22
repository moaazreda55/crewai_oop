#!/usr/bin/env python
import sys
import warnings
from my_crew2.crew import MyCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

# print(MyCrew().Keyword_task())

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'Deep Learning'
    }
    
    try:

        MyCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


run()

