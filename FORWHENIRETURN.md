# NOTES FOR AFTER ONE WEEK BREAK

- Setup a FastAPI server
- lt_memory needs to be finalized because it does not have LLM integration
- Automation engine needs a total rebuild
- Tools need to be gone through one-by-one to bring them all in line with most-current standards
- Remove sentence-transformers and replace it with onnx because I already have it set up for some tasks but not in the main.py
- Go through the docs/ folder and clear out junk that isn't relevant. it has become a catchall for markdown files.
- Remove all of the old classifier examples and generate new ones (the generator needs to correctly name the files so that MIRA recognizes them)
- Create a modern sample_tool.py
- Explore the ability to have "topics" mean something (e.g., "email" is a topic, "recipe" is a topic, etc.)
- Marry topics/workflows to send data over the link as an aside to the actual streamed message (e.g., show inbox count when in email context and clear it when changing topics)