---
name: ODAI Documentation Guidelines
description: Documentation style rules and checklist for the ODAI SDK.
---

## General Format

1. **Use Doxygen-style comments**: Always use `///` (triple slash) for documentation comments, not `//` or `/* */`

2. **First line is a brief summary**: The first line should be a concise, one-line description of what the function/class/struct does

3. **Multi-line descriptions**: If more detail is needed, add additional lines after the first line to explain behavior, edge cases, or important notes

## Function Documentation

### Required Elements

1. **Brief description**: Start with a clear, concise description of the function's purpose

2. **Parameter documentation**: Use `@param` tags for each parameter:
   - Format: `/// @param parameter_name Description of the parameter`
   - Include type information if not obvious from the signature
   - Mention if the parameter is modified in place
   - Note if the parameter is optional or has special constraints

3. **Return value documentation**: Use `@return` tag:
   - Format: `/// @return Description of return value`
   - Always mention error conditions (e.g., "or -1 on error", "or nullptr on error", "or empty vector on error")
   - Specify what the return value represents (e.g., "Total number of tokens generated", "true if successful")

### Additional Guidelines

- **Mention edge cases**: Document important behaviors like "If the same model is already loaded, only updates the configuration"
- **Explain complex behavior**: For functions with non-obvious behavior, provide multi-line explanations
- **Note side effects**: Mention if parameters are modified in place or if the function has side effects
- **Reserved parameters**: Note if a parameter is "currently unused, reserved for future use"
- **Unimplemented functions**: If a function is declared but not yet implemented, add a "ToDo: Implementation not yet defined." note in the documentation
- **Avoid implementation details**: Do not expose unnecessary implementation details in documentation. Focus on what the function does from the user's perspective, not how it's implemented internally


## Checklist

When documenting a function, ensure:
- [ ] Brief description on first line
- [ ] All parameters documented with `@param`
- [ ] Return value documented with `@return` including error conditions
- [ ] Edge cases and special behaviors mentioned
- [ ] Side effects documented if any
- [ ] Complex behavior explained in additional lines
