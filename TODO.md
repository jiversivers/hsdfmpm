- [ ] Validate the LUT on its own data (model HB/Ci -> fit modelled)
- [ ] Correct correction factors
  1. In MATLAB code, figure out what actualy scatter was (not reduced)
  2. Model (non-reduced) scattering with MCLUT
  3. Find correction factor
  4. Re-fit
- [ ] Validate LUT on food coloring
- [ ] Determine registration method
- [ ] Check that laser/PMT calibration is happening in ORR object
- [ ] Generate workflow for HSDFM processing
- [ ] Plot phasor validation
- [ ] Make photon_canon.contrib.bio vectorizable (i.e. mu calculations should work if a and b are arrays)