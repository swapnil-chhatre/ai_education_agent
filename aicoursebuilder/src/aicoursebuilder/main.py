#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()

import sys
import warnings

from crew import Aicoursebuilder






warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs'
    }
    Aicoursebuilder().crew().kickoff(inputs=inputs)
    

run()
