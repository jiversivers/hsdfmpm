- [x] Validate the LUT on its own data (model HB/Ci -> fit modelled)
- [x] Correct correction factors
  1. In MATLAB code, figure out what actually scatter was (not reduced)
  2. Model (non-reduced) scattering with MCLUT
  3. Find correction factor
  4. Re-fit
- [x] Validate LUT on food coloring
- [x] Determine registration method
- [x] Check that laser/PMT calibration is happening in ORR object
- [x] Generate workflow for HSDFM processing
- [x] Plot phasor validation
- [ ] Make photon_canon.contrib.bio vectorizable (i.e. mu calculations should work if a and b are arrays)
- [ ] Document this package (in progress)
- [x] Add $\chi^2_\nu$ score for phasor fit line (or generic score function with default)
- [ ] Add tests for new components