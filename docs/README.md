## BMTK documentation, guides, and examples

### DPointNet user and configuration guides

- [DPointNet user guide](autodocs/source/dpointnet_guide.rst)
- [Performance configuration and precision](dpointnet_parity.md)
- [Variable-batch acceleration and device Poisson sampling](dpointnet_variable_batch.md)
- [Automatic alpha-basis fitting](dpointnet_alpha_basis.md)
- [Per-device LGN input generation](dpointnet_lgn_pipeline.md)
- [Pascal GPU compatibility and limitations](dpointnet_pascal.md)
- [Initial-state input recovery diagnostics](dpointnet_input_recovery.md)

These Markdown supplements can be read directly on GitHub. They are not currently
included in the [published user guide](https://alleninstitute.github.io/bmtk/dpointnet_guide.html)
navigation. Consult documentation from the same revision as your installed BMTK.

### DPointNet developer notes

- [Startup preprocessing and ordering invariants](dpointnet_startup.md)

#### directory structure

- autodocs/ - scripts and pages for the generation of github-pages html files.
- tutorial/ - Tutorials and guides for using bmtk and its different parts. **New users should start here**.
- examples/ - Various examples of how to build networks, run various simulations, and plot their results. A good place to start for users wanting a quick and dirty introduction (warning: many of these examples are not as well documented as the tutorials).
