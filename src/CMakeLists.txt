add_executable("dune-diffusionfem" dune-diffusionfem.cc poissonPDE.hh ellipticOperator.hh problemInterface.hh FEMscheme.hh rhs.hh nonlinearModel.hh)
target_link_dune_default_libraries("dune-diffusionfem")
target_compile_definitions( dune-diffusionfem PRIVATE YASPGRID;GRIDDIM=3;WORLDDIM=3;POLORDER=1 )