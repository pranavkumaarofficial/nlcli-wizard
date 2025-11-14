---
license: apache-2.0
task_categories:
- question-answering
language:
- en
size_categories:
- 10K<n<100K
---
# Natural Language to Docker Command Dataset

This dataset is designed to translate natural language instructions into Docker commands. It contains mappings of textual phrases to corresponding Docker commands, aiding in the development of models capable of understanding and translating user requests into executable Docker instructions.

## Dataset Format

Each entry in the dataset consists of a JSON object with the following keys:

- `input`: The natural language phrase.
- `instruction`: A static field indicating the task to translate the phrase into a Docker command.
- `output`: The corresponding Docker command.

### Example Entry

```json
{
  "input": "Can you show me the digests of all the available Docker images?",
  "instruction": "translate this sentence in docker command",
  "output": "docker images --digests"
}
```

## Usage

This dataset can be utilized to train and evaluate models for a variety of applications including, but not limited to, Natural Language Processing (NLP), Command Line Interface (CLI) automation, and educational tools for Docker.

## Commands coverage

- docker ps
- docker images
- docker stop
- docker kill
- docker login

## Contributing

We welcome contributions to improve this dataset. Please feel free to open a Pull Request or an Issue to discuss potential improvements, bug fixes, or other changes.