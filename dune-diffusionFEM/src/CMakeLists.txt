add_executable("dune-diffusionFEM" dune-diffusionFEM.cc poissonPDE.hh ellipticOperator.hh problemInterface.hh dirichletConstraints.hh FEMscheme.hh rhs.hh nonlinearModel.hh)
target_link_dune_default_libraries("dune-diffusionFEM")
