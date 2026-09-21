"""mambo jambo"""


import pytest


@pytest.fixture
def sample_python() -> str:
    return '''\
        def add(a, b):
            return a + b


        def subtract(a, b):
            return a - b


        class Calculator:
            def multiply(self, a, b):
                return a * b

            def divide(self, a, b):
                return a / b
    '''


@pytest.fixture
def sample_markdown() -> str:
    return '''\
        # Title

        Some introductory paragraph explaining the topic.

        ## Section One

        Content for section one goes here, with a bit more detail.

        ## Section Two

        Content for section two, also with some detail to pad it out.
    '''


@pytest.fixture
def sample_txt() -> str:
    return '''\
        This is a plain text file.
        It has a few lines of unstructured content.
        Nothing fancy, just prose.
    '''
