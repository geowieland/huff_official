# Contributing to huff

Thank you for your interest in contributing to `huff`.

`huff` is an open-source Python package for market area analysis, including Huff and MCI models, spatial accessibility analysis, and related GIS and network analysis tools. The particular focus is on the fact that the package covers the entire workflow of a market area analysis. This expressly includes the calibration of market area models (adjusting weighting parameters to empirical data), which constitutes a second key focus of the package.

## Reporting issues

Bug reports, questions, and feature requests are welcome.

Please use the [GitHub Issues](https://github.com/geowieland/huff_official/issues) page to report:

- bugs or unexpected behavior,
- documentation issues,
- feature requests,
- questions about the use of the package.

When reporting a bug, please provide a minimal reproducible example where possible, including the `huff` version and relevant Python/environment information.

## Contributing code

Contributions via pull requests are welcome.

Before submitting a pull request:

1. Fork the repository and create a separate branch for your changes.
2. Keep changes focused on a single feature, bug fix, or documentation improvement.
3. Follow the existing code structure and style.
4. Update the documentation or examples when appropriate.
5. Add or update tests where appropriate.
6. Make sure that existing functionality is not unintentionally changed.

Pull requests should include a short description of the changes and their motivation.

## Documentation

Improvements to the documentation, examples, and methodological explanations are welcome.

Documentation changes should be consistent with the existing terminology and structure of the project.

## Development

The package can be installed directly from the repository:

```bash
pip install git+https://github.com/geowieland/huff_official.git
```

For development, clone the repository and install the package locally:

```bash
git clone https://github.com/geowieland/huff_official.git
cd huff_official
pip install -e .
```

The [examples/](https://github.com/geowieland/huff_official/tree/main/examples) directory contains examples of using the package.

## Scientific contributions

Contributions related to the implementation or extension of market area, Huff/MCI, accessibility, or GIS methods should include appropriate references to the underlying scientific literature where relevant.

Please describe methodological changes clearly so that their scientific purpose and implementation can be reviewed.

If you use the `huff` Python package, please [cite the software](https://github.com/geowieland/huff_official/blob/main/CITATION.cff).

## Code of conduct

Please keep discussions and contributions respectful, constructive, and focused on improving the software.

## License

By contributing to this repository, you agree that your contributions will be licensed under the MIT License used by `huff`.