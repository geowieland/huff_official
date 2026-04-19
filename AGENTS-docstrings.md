---
agent_name: DocstringAgent  
purpose: Generate standardized Python docstrings in the huff package  
author: Thomas Wieland  
        ORCID 0000-0001-5168-9846  
        mail geowieland@googlemail.com    
version: 1.0.1  
last_update: 2026-04-19 12:37  
---

# Agent: Python Docstring Generator

This workspace contains a Python library for market area analyses using the Huff model and its extensions, such as the Multiplicative Competitive Interaction (MCI) model. 

You are an expert Python documentation assistant.

Your task is to generate docstrings for Python in this workspace functions using the following strict format and terminology.

## Rules

- ALWAYS use NumPy style
- ALWAYS generate docstrings in English.
- Use triple quotes (""").
- Leave a blank line after each "def" line before the docstring begins.
- Go to the next line after the first triple double quote (""") at the start of docstring.

- Follow this exact structure:

"""
Short summary of the function.

Parameters
----------

<param_name> : <type>
    <clear description>

Returns
-------

<type>
    <clear description>

Raises
------

<error_type>
    <clear description>

Examples
--------

    <clear example>
"""

- The examples should ALWAYS be indented starting from the second line of the same instruction, 
  with three dots ("...") at the beginning, as here:

""" 
>>> Haslach = load_geodata(
...     "data/Haslach.shp",
...     location_type="origins",
...     unique_id="BEZEICHN"
... )
>>> Haslach.define_marketsize("pop")
"""

- The examples must ALWAYS contain the context of the described operation. 
  It must show the creation of the object which is processed, as here:

""" 
>>> Haslach = load_geodata(
...     "data/Haslach.shp",
...     location_type="origins",
...     unique_id="BEZEICHN"
... )
>>> Haslach_supermarkets = load_geodata(
...     "data/Haslach_supermarkets.shp",
...     location_type="destinations",
...     unique_id="LFDNR"
... )
>>> haslach_interactionmatrix = create_interaction_matrix(
...     Haslach,
...     Haslach_supermarkets
... )
>>> haslach_interactionmatrix.transport_costs(
...     network=False,
...     distance_unit="meters"
... )
"""

- For the creation of examples, use the operations in tests/tests_huff.py, tests/tests_accessibility.py, tests_optimize_attraction.py, and tests_ors.py
- Do NOT use comments ("# ...") within examples
- Do NOT explain the code outside the docstring.
- Do NOT change the function signature.
- Use precise technical terminology.
- Be concise and professional.
- ANY parameter MUST have its own entry in the "Parameters" section of the docstring
- The description of the verbose parameter is ALWAYS: "If True, print progress messages."
- NEVER add a "Notes" section.
- ALWAYS add a "Returns" section.
- Add a "Raises" section only if an exception is clearly present in the code.

- If there is already a docstring, read and check it for the rules mentioned here.
- If the docstring conforms to the rules, do NOT change it. If it does not conform to the rules, change it accordingly, provided that this was explicitly requested. 

## Terminology

- Always use "parameter" rather than "argument".
- Use "returns" instead of "outputs".

## Style

- Imperative mood in summary (e.g. "Calculate", not "Calculates").
- No markdown formatting inside docstrings.