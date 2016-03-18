add_executable("dune-diffusionfem" dune-diffusionfem.cc poissonPDE.hh ellipticOperator.hh problemInterface.hh FEMscheme.hh rhs.hh nonlinearModel.hh)
target_link_dune_default_libraries("dune-diffusionfem")
