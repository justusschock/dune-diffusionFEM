add_executable("dune-diffusionFEM" dune-diffusionFEM.cc diffusionPDE.hh)
target_link_dune_default_libraries("dune-diffusionFEM")
