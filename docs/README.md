## BMTK documentation

### For users

Start with the [DPointNet guide](autodocs/source/dpointnet_guide.rst) for a
complete small simulation and configuration overview. Continue with
[training](autodocs/source/dpointnet_training.rst) and
[defaults and migration](autodocs/source/dpointnet_training_standards.rst).
Define your own objectives and training hooks with
[custom losses and callbacks](autodocs/source/dpointnet_custom_training.rst).
The runnable [300-neuron example](../examples/dpointnet_all2all/) includes
network/component files and target rates. Optional visual inputs, network
import, performance and troubleshooting are linked from the guide.

Use documentation from the same revision as your installed BMTK. General
BMTK notebooks are in [tutorial/](tutorial/); examples for the other simulators
are in the repository's [examples/](../examples/).

### For documentation contributors

The Sphinx website uses reStructuredText pages in
[autodocs/source/](autodocs/source/). Add pages to the appropriate toctree.
Keep runnable user workflows separate from
[DPointNet implementation notes](autodocs/source/dpointnet_development.rst).

Install the repository's `doc_requirements.txt`, then run `make html` from
`docs/autodocs`. To render without executing tutorial notebooks:

```bash
make html SPHINXOPTS="-D nbsphinx_execute=never"
```

The build copies tutorial notebooks and static assets into its source tree.
Use a disposable copy when validating without generated source changes.
Inspect the rendered navigation, tables and examples as well as build warnings.
