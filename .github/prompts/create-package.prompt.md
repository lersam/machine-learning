---
name: create-package
description: 'Create a new Python package from a package name input'
argument-hint: 'packageName=my_new_package'
agent: agent
---

# Create Python Package

Create a new Python package in this repository.

Package name: ${input:packageName:my_new_package}

Additional request details from chat: ${input:details:Optional target folder or extra requirements}

Follow the repository standards in [copilot-instructions](../copilot-instructions.md) and [agents.md](../../agents.md).

Requirements:

- Use the provided package name as the primary input.
- If the package name is not a valid Python package name, normalize it to `snake_case` and explain the normalization.
- Unless the user explicitly asks for another location, create the package in the most appropriate existing Python source area for this repository.
- Inspect nearby packages before editing so the new package matches local structure and naming patterns.
- Create the package directory and add an `__init__.py` file.
- Add the smallest useful scaffold needed for the package to exist cleanly.
- Do not overwrite an existing package. If the target package already exists, stop and explain what already exists.
- Keep changes minimal and focused on the requested package.
- Summarize the files you created and the final package path.

If the user provides extra requirements in chat, apply them as long as they do not conflict with the repository standards.