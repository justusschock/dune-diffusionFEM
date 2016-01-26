add_executable("dune-diffusionPDELab" dune-diffusionPDELab.cc stationaryDiffusion.hh poisson.hh PDETestSuite.h instationaryDiffusion.h l2interpolationerror.h)
target_link_dune_default_libraries("dune-diffusionPDELab")

target_link_libraries(dune-diffusionPDELab cpptest)
