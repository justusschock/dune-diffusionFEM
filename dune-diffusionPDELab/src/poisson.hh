
#ifndef DUNE_TEST_POISSON_HH
#define DUNE_TEST_POISSON_HH

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include<iostream>
#include<vector>
#include<map>
#include<string>
#include<dune/common/parallel/mpihelper.hh>
#include<dune/common/exceptions.hh>
#include<dune/common/fvector.hh>
#include<dune/common/float_cmp.hh>
#include<dune/common/static_assert.hh>
#include<dune/grid/yaspgrid.hh>
#include<dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/operators.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/preconditioners.hh>
#include<dune/istl/io.hh>

#include<dune/pdelab/finiteelementmap/pkfem.hh>
#include<dune/pdelab/finiteelementmap/qkfem.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#include<dune/pdelab/gridfunctionspace/interpolate.hh>
#include<dune/pdelab/constraints/common/constraints.hh>
#include<dune/pdelab/constraints/common/constraintsparameters.hh>
#include<dune/pdelab/constraints/conforming.hh>
#include<dune/pdelab/constraints/hangingnode.hh>
#include<dune/pdelab/common/function.hh>
#include<dune/pdelab/common/vtkexport.hh>
#include<dune/pdelab/backend/istlvectorbackend.hh>
#include<dune/pdelab/backend/istl/bcrsmatrixbackend.hh>
#include<dune/pdelab/backend/istlmatrixbackend.hh>
#include<dune/pdelab/backend/istlsolverbackend.hh>
#include<dune/pdelab/backend/seqistlsolverbackend.hh>
#include<dune/pdelab/localoperator/laplacedirichletp12d.hh>
#include<dune/pdelab/localoperator/poisson.hh>
#include<dune/pdelab/gridoperator/gridoperator.hh>
#include<dune/pdelab/stationary/linearproblem.hh>
#include<dune/pdelab/gridfunctionspace/vtk.hh>



/*
  HANGING_NODES_REFINEMENT is macro used to switch on hanging nodes tests.
  It is set in "Makefile.am" to generate the executable 'poisson_HN'.
*/

//===============================================================
//===============================================================
// Solve the Poisson equation
//           - \Delta u = f in \Omega,
//                    u = g on \partial\Omega_D
//  -\nabla u \cdot \nu = j on \partial\Omega_N
//===============================================================
//===============================================================

//===============================================================
// Define parameter functions f,g,j and \partial\Omega_D/N
//===============================================================



// function for defining the source term
template<typename GV, typename RF>
class F
        : public Dune::PDELab::AnalyticGridFunctionBase<Dune::PDELab::AnalyticGridFunctionTraits<GV,RF,1>,
                F<GV,RF> >
{
public:
    typedef Dune::PDELab::AnalyticGridFunctionTraits<GV,RF,1> Traits;
    typedef Dune::PDELab::AnalyticGridFunctionBase<Traits,F<GV,RF> > BaseT;

    F (const GV& gv) : BaseT(gv) {}
    inline void evaluateGlobal (const typename Traits::DomainType& x,
                                typename Traits::RangeType& y) const
    {
        if (x[0]>0.25 && x[0]<0.375 && x[1]>0.25 && x[1]<0.375)
            y = 50.0;
        else
            y = 0.0;
        y=0;
    }
};



// constraints parameter class for selecting boundary condition type
class BCTypeParam
        : public Dune::PDELab::DirichletConstraintsParameters /*@\label{bcp:base}@*/
{
public:

    template<typename I>
    bool isDirichlet(
            const I & intersection,   /*@\label{bcp:name}@*/
            const Dune::FieldVector<typename I::ctype, I::dimension-1> & coord
    ) const
    {

        Dune::FieldVector<typename I::ctype, I::dimension>
                xg = intersection.geometry().global( coord );

        if( xg[1]<1E-6 || xg[1]>1.0-1E-6 )
            return false; // Neumann b.c.
        else if( xg[0]>1.0-1E-6 && xg[1]>0.5+1E-6 )
            return false; // Neumann b.c.
        else
            return true;  // Dirichlet b.c. on all other boundaries
    }

    template<typename I>
    bool isNeumann(
            const I & intersection,   /*@\label{bcp:name}@*/
            const Dune::FieldVector<typename I::ctype, I::dimension-1> & coord
    ) const
    {
        return !isDirichlet(intersection,coord);
    }

};

// function for Dirichlet boundary conditions and initialization
template<typename GV, typename RF>
class G
        : public Dune::PDELab::AnalyticGridFunctionBase<Dune::PDELab::AnalyticGridFunctionTraits<GV,RF,1>,
                G<GV,RF> >
{
public:
    typedef Dune::PDELab::AnalyticGridFunctionTraits<GV,RF,1> Traits;
    typedef Dune::PDELab::AnalyticGridFunctionBase<Traits,G<GV,RF> > BaseT;

    G (const GV& gv) : BaseT(gv) {}
    inline void evaluateGlobal (const typename Traits::DomainType& x,
                                typename Traits::RangeType& y) const
    {
        typename Traits::DomainType center;
        for (int i=0; i<GV::dimension; i++) center[i] = 0.5;
        center -= x;
        y = exp(-center.two_norm2());
    }
};

// function for defining the flux boundary condition
template<typename GV, typename RF>
class J
        : public Dune::PDELab::AnalyticGridFunctionBase<Dune::PDELab::AnalyticGridFunctionTraits<GV,RF,1>,
                J<GV,RF> >
{
public:
    typedef Dune::PDELab::AnalyticGridFunctionTraits<GV,RF,1> Traits;
    typedef Dune::PDELab::AnalyticGridFunctionBase<Traits,J<GV,RF> > BaseT;

    J (const GV& gv) : BaseT(gv) {}
    inline void evaluateGlobal (const typename Traits::DomainType& x,
                                typename Traits::RangeType& y) const
    {
        if (x[1]<1E-6 || x[1]>1.0-1E-6)
        {
            y = 0;
            return;
        }
        if (x[0]>1.0-1E-6 && x[1]>0.5+1E-6)
        {
            y = -5.0;
            return;
        }
    }
};







//===============================================================
// Problem setup and solution
//===============================================================

// generate a P1 function and output it
template<typename GV, typename FEM, typename BCTYPE, typename CON>
void poisson_driver(const GV& gv,
                    const FEM& fem,
                    std::string filename,
                    const BCTYPE& bctype,    // boundary condition type
                    bool hanging_nodes,
                    int q,  // quadrature order
                    const CON& con = CON())
{
    // constants and types
    typedef typename FEM::Traits::FiniteElementType::Traits::
    LocalBasisType::Traits::RangeFieldType R;

    // make grid function space
    typedef Dune::PDELab::ISTLVectorBackend<> VBE;
    typedef Dune::PDELab::GridFunctionSpace<GV,FEM,CON,VBE> GFS;
    GFS gfs(gv,fem,con);
    gfs.name("poisson solution");

    // make constraints map and initialize it from a function
    typedef typename GFS::template ConstraintsContainer<R>::Type C;
    C cg;
    cg.clear();

    Dune::PDELab::constraints(bctype,gfs,cg);

    // make grid operator
    typedef F<GV,R> FType;
    FType f(gv);
    typedef J<GV,R> JType;
    JType j(gv);
    typedef Dune::PDELab::Poisson<FType,BCTypeParam,JType> LOP;
    LOP lop(f,bctype,j,q);

    typedef Dune::PDELab::istl::BCRSMatrixBackend<> MBE;
    MBE mbe(45); // Maximal number of nonzeroes per row can be cross-checked by patternStatistics().

    typedef Dune::PDELab::GridOperator<GFS,GFS,LOP,MBE,R,R,R,C,C> GO;
    GO go(gfs,cg,gfs,cg,lop,mbe);

    // make coefficent Vector and initialize it from a function
    typedef typename GO::Traits::Domain V;
    V x0(gfs);
    x0 = 0.0;
    typedef G<GV,R> GType;
    GType g(gv);
    Dune::PDELab::interpolate(g,gfs,x0);

    // Choose ISTL Solver Backend
    typedef Dune::PDELab::ISTLBackend_SEQ_CG_SSOR LS;
    //typedef Dune::PDELab::ISTLBackend_SEQ_CG_AMG_SSOR LS;
    //typedef Dune::PDELab::ISTLBackend_SEQ_CG_ILU0 LS;
    LS ls(5000,2);

    typedef Dune::PDELab::StationaryLinearProblemSolver<GO,LS,V> SLP;
    SLP slp(go,ls,x0,1e-12);
    slp.setHangingNodeModifications(hanging_nodes);
    slp.apply();

    // make discrete function object
    Dune::SubsamplingVTKWriter<GV> vtkwriter( gv, 1 );
    //Dune::VTKWriter<GV> vtkwriter(gv,Dune::VTK::conforming);
    Dune::PDELab::addSolutionToVTKWriter(vtkwriter,gfs,x0);
    vtkwriter.write(filename,Dune::VTK::ascii);
}


//===============================================================
//===============================================================
// Solve the Poisson equation
//---------------------------------------------------------------
//           - \Delta u = f in \Omega,
//                    u = g on \partial\Omega_D
//  -\nabla u \cdot \nu = j on \partial\Omega_N
//===============================================================
//===============================================================

template<const int dim, const int k>
int solvePoissonPDE(int maxlevel, Dune::YaspGrid<dim>& grid)
{
    try {

        grid.globalRefine(maxlevel - 1);

        // get view
        using GV = typename Dune::YaspGrid<dim>::LeafGridView;
        const GV &gv = grid.leafGridView();


        const int q = 2 * k;
        // make finite element map
        using DF = typename GV::Grid::ctype;
        using FEM = Dune::PDELab::QkLocalFiniteElementMap<GV, DF, double, k>;
        FEM fem(gv);

        BCTypeParam bctype;
        // solve problem
        using Constraints = Dune::PDELab::ConformingDirichletConstraints;
        poisson_driver<GV, FEM, BCTypeParam, Constraints>(gv, fem, "poisson_yasp", bctype, false, q);


        return 0;
    }
    catch (Dune::Exception &e){
        std::cerr << "Dune reported error: " << e << std::endl;
        return false;
    }
    catch (std::string &e){
        std::cerr << "An error has been detected: " << e << std::endl;
        return false;
    }
    catch (std::exception &e){
        std::cerr << "STL reported error: " << e.what() << std::endl;
    }
    catch (...){
        std::cerr << "Unknown exception thrown!" << std::endl;
        return false;
    }


}


#endif //DUNE_TEST_POISSON_HH