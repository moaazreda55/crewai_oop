import streamlit as st
import warnings
from my_crew2.crew import MyCrew


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


st.title("Generating Articles and Producing final report")


def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'Deep Learning'
    }
    
    try:

        result = MyCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

    for task in result.tasks_output:

        for line in task.raw.split("\n"):

            st.markdown(line.strip())
    
        st.markdown("*"*26 + " Task End " + "*"*26)


run()




