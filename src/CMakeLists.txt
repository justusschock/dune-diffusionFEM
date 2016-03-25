add_executable("dune-diffusionfem" dune-diffusionfem.cc poissonPDE.hh ellipticOperator.hh problemInterface.hh FEMscheme.hh rhs.hh nonlinearModel.hh elementdata.hh)
target_link_dune_default_libraries("dune-diffusionfem")
target_compile_definitions( dune-diffusionfem PRIVATE YASPGRID;GRIDDIM=2;WORLDDIM=2;POLORDER=1 )
