add_executable("dune-diffusionFEM" dune-diffusionFEM.cc diffusionPDE.hh ellipticOperator.hh problemInterface.hh dirichletConstraints.hh FEMscheme.hh rhs.hh diffusionModel.hh)
target_link_dune_default_libraries("dune-diffusionFEM")
