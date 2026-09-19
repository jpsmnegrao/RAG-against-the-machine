*This project has been created as part of the 42 curriculum by jpaim-so.*


A README.md file must be provided at the root of your Git repository. Its purpose is
to allow anyone unfamiliar with the project (peers, staff, recruiters, etc.) to quickly
understand what the project is about, how to run it, and where to find more information
on the topic.

The README.md must include at least:
• The very first line must be italicized and read: This project has been created as part
of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].
• A “Description” section that clearly presents the project, including its goal and a
brief overview.
• An “Instructions” section containing any relevant information about compilation,
installation, and/or execution.
• A “Resources” section listing classic references related to the topic (documentation, articles, tutorials, etc.), as well as a description of how AI was used —
specifying for which tasks and which parts of the project.

➠ Additional sections may be required depending on the project (e.g., usage
examples, feature list, technical choices, etc.).
Any required additions will be explicitly listed below.
For this project, the README.md must also include:
• System architecture: Describe your RAG pipeline components and how they
interact
• Chunking strategy: Explain your approach to document segmentation
• Retrieval method: Detail the retrieval algorithm and ranking mechanism
21
RAG against the machine Will you answer my questions?
• Performance analysis: Discuss recall@k scores and system performance
• Design decisions: Explain key implementation choices
• Challenges faced: Document difficulties encountered and solutions
• Example usage: Provide clear examples of running your system


# RAG-against-the-machine


## Description

Mambo jambo.


## Instructions

### Requirements

- Python 3.10+;
- `uv`;
- The required `llm_sdk` package and model environment.

#### NOTE

Disk quota / cache configuration

The project is configured in pyproject.toml to store the uv cache locally in
.uv-cache. This is intentional because 42 machines have strict disk-quota
limits on the home partition.

The project can therefore be cloned to any location and installed with:

`make install`

No additional configuration is required on a personal computer.


### Installation

Install the project dependencies and set up the virtual environment with:

`make`

or:

`make install`

### Run

`make run`

### Debugging

Run the application with Python's debugger using:

`make debug`

### Testing

Run the complete pytest test suite with:

`make test`

### Linting and Type Checking

Run the standard code-quality checks with:

`make lint`

Executes the commands flake8 . and mypy . --warn-return-any
--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs
--check-untyped-defs

For stricter type checking:

`make lint-strict`

Which executes the commands flake8 . and mypy . --strict.

### Cleaning

Remove generated caches and Python bytecode with:

`make clean`

To additionally remove the virtual environment and generated output files:

`make fclean`

### Project structure

```
data/datasets/AnsweredQuestions
├── dataset_code_public.json
├── dataset_docs_public.json

data/datasets/UnansweredQuestions
├── dataset_code_public.json
├── dataset_docs_public.json

src/
├── __init__.py
├── __main__.py
├── schemas.py
├── indexing.py

```

## Resources

### Documentation

- [Pydantic documentation](https://docs.pydantic.dev/) — used throughout for
  schema validation.
- [Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)


### AI Usage

AI was used throughout development as a conversational aid for:


## Additional sections:

### System architecture

### Chunking strategy

### Retrieval method

### Performance analysis

### Design decisions

### Challenges faced

### Example usage
