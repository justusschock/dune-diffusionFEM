#ifndef DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH
#define DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH

#include <cassert>
#include <cmath>
// include output
#include <dune/fem/io/file/dataoutput.hh>

// iostream includes
#include <iostream>

// include grid part
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>

// include output
#include <dune/fem/io/file/dataoutput.hh>
#include <dune/fem/io/file/vtkio.hh>

// include header of elliptic solver
#include "FEMscheme.hh"

#include "problemInterface.hh"

const bool graphics = true;


//general form of Robin b.c.: a*u + b* du/dn = g on boundary of omega

// -laplace u + u = f with Dirichlet and Neumann boundary conditions on domain [0,1]^d (now: Robin b.c.)
// Exact solution is u(x_1,...,x_d) = cos(2*pi*x_1)...cos(2*pi*x_d)
template <class FunctionSpace>
class CosinusProduct : public ProblemInterface < FunctionSpace >
{
    typedef ProblemInterface < FunctionSpace >  BaseType;
public:
    typedef typename BaseType :: RangeType            RangeType;
    typedef typename BaseType :: DomainType           DomainType;
    typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
    typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

    enum { dimRange  = BaseType :: dimRange };
    enum { dimDomain = BaseType :: dimDomain };

    //! the right hand side data (default = 0)
    virtual void f(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 4*dimDomain*(M_PI*M_PI);
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::cos( 2*M_PI*x[ i ] );
        RangeType uVal;
        u(x,uVal);
        phi += uVal;
    }

    //! the exact solution
    virtual void u(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 1;
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::cos( 2*M_PI*x[ i ] );
    }

    //! the jacobian of the exact solution
    virtual void uJacobian(const DomainType& x,
                           JacobianRangeType& ret) const
    {
        for( int r = 0; r < dimRange; ++ r )
        {
            for( int i = 0; i < dimDomain; ++i )
            {
                ret[ r ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] );
                for( int j = 1; j < dimDomain; ++j )
                    ret[ r ][ i ] *= std::cos( 2*M_PI*x[ (i+j)%dimDomain ] );
            }
        }
    }

    //! mass coefficient has to be 1 for this problem
    virtual void m(const DomainType& x, RangeType &m) const
    {
        m = RangeType(1);
    }

    virtual void alpha(const DomainType& x, RangeType &a) const
    {
        a = RangeType(0.5);
    }
    //! the Dirichlet boundary data (default calls u)
    virtual void g(const DomainType& x,
                   RangeType& value) const
    {
        u(x,value);
    }
    virtual bool hasDirichletBoundary () const
    {
        return true ;
    }
    virtual bool hasNeumanBoundary () const
    {
        return true ;
    }
    virtual bool isDirichletPoint( const DomainType& x ) const
    {
        // all boundaries except the x=0 plane are Dirichlet
        return (std::abs(x[0])>1e-8);

    }
    virtual void n(const DomainType& x,
                   RangeType& value) const
    {
        u(x,value);
        value *= 0.5;
        JacobianRangeType jac;
        uJacobian(x,jac);
        value[0] -= jac[0][0];
    }
};

// -laplace u = f with Dirichlet and Neumann boundary conditions on domain [0,1]^d (now: Robin b.c.)
// Exact solution is u(x_1,...,x_d) = sin(2*pi*x_1)...sin(2*pi*x_d)
template <class FunctionSpace>
class SinusProduct : public ProblemInterface < FunctionSpace >
{
    typedef ProblemInterface < FunctionSpace >  BaseType;
public:
    typedef typename BaseType :: RangeType            RangeType;
    typedef typename BaseType :: DomainType           DomainType;
    typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
    typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

    enum { dimRange  = BaseType :: dimRange };
    enum { dimDomain = BaseType :: dimDomain };

    //! the right hand side data (default = 0)
    virtual void f(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 4*dimDomain*(M_PI*M_PI);
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::sin( 2*M_PI*x[ i ] );
    }

    //! the exact solution
    virtual void u(const DomainType& x,
                   RangeType& phi) const
    {
        phi = 1;
        for( int i = 0; i < dimDomain; ++i )
            phi *= std::sin( 2*M_PI*x[ i ] );
        phi[0] += x[0]*x[0]-x[1]*x[1]+x[0]*x[1];
        // phi[0] += x[0]*x[1];
        // phi[0] += 0.5;
    }

    //! the jacobian of the exact solution
    virtual void uJacobian(const DomainType& x,
                           JacobianRangeType& ret) const
    {
        for( int r = 0; r < dimRange; ++ r )
        {
            for( int i = 0; i < dimDomain; ++i )
            {
                ret[ r ][ i ] = 2*M_PI*std::cos( 2*M_PI*x[ i ] );
                for( int j = 1; j < dimDomain; ++j )
                    ret[ r ][ i ] *= std::sin( 2*M_PI*x[ (i+j)%dimDomain ] );
            }
        }
        ret[0][0] +=  2.*x[0]+x[1];
        ret[0][1] += -2.*x[1]+x[0];
        // ret[0][0] += x[1];
        // ret[0][1] += x[0];
    }
    //! mass coefficient has to be 1 for this problem
    virtual void m(const DomainType& x, RangeType &m) const
    {
        m = RangeType(1);
    }

    //DomainType = Type of input variable (e.g const double)
    //RangeType = Type of output variable (E.g. double)
    virtual void alpha(const DomainType& x, RangeType &a) const
    {
        a = RangeType(0.5);
    }
    //! the Dirichlet boundary data (default calls u)
    virtual void g(const DomainType& x,
                   RangeType& value) const
    {
        u(x,value);
    }
    virtual bool hasDirichletBoundary () const
    {
        return true ;
    }
    virtual bool hasNeumanBoundary () const
    {
        return true ;
    }
    virtual bool isDirichletPoint( const DomainType& x ) const
    {
        // all boundaries except the x=0 plane are Dirichlet
        return (std::abs(x[0])>1e-8);

    }
    virtual void n(const DomainType& x,
                   RangeType& value) const
    {
        u(x,value);
        value *= 0.5;
        JacobianRangeType jac;
        uJacobian(x,jac);
        value[0] -= jac[0][0];
    }
};


// assemble-solve-estimate-mark-refine-IO-error-doitagain
template <class HGridType>
double algorithm ( HGridType &grid, int step, const int problemNumber )
{
    // we want to solve the problem on the leaf elements of the grid
    typedef Dune::Fem::AdaptiveLeafGridPart< HGridType > GridPartType;
    GridPartType gridPart(grid);

    // use a scalar function space
    typedef Dune::Fem::FunctionSpace< double, double,
            HGridType::dimensionworld, 1 > FunctionSpaceType;
    // type of the mathematical model used
    typedef nonlinearModel< FunctionSpaceType, GridPartType > ModelType;

    typedef typename ModelType::ProblemType ProblemType ;
    ProblemType* problemPtr = 0 ;
    std::stringstream fullname;
    fullname << "poisson_";

    switch ( problemNumber )
    {
        case 0:
            problemPtr = new CosinusProduct< FunctionSpaceType > ();
            fullname << "cos_problem_step_" << step;
            break ;
        case 1:
            problemPtr = new SinusProduct< FunctionSpaceType > ();
            fullname << "sin_problem_step_" << step;
            break ;
        default:
            problemPtr = new CosinusProduct< FunctionSpaceType > ();
            fullname << "cos_problem_step_" << step;
    }
    assert( problemPtr );
    ProblemType& problem = *problemPtr ;

    // implicit model for left hand side
    ModelType implicitModel( problem, gridPart );

    // poisson solver
    typedef FemScheme< ModelType > SchemeType;
    SchemeType scheme( gridPart, implicitModel );

    typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPartType > GridExactSolutionType;
    GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );
    //! input/output tuple and setup datawritter
    typedef Dune::tuple< const typename SchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
    typedef Dune::Fem::DataOutput< HGridType, IOTupleType > DataOutputType;
    IOTupleType ioTuple( &(scheme.solution()), &gridExactSolution) ; // tuple with pointers
    DataOutputType dataOutput( grid, ioTuple, DataOutputParameters( step ) );

    // setup the right hand side
    scheme.prepare();
    // solve once (assemble matrix)
    scheme.solve(true);

    // write initial solve
    dataOutput.write("~/ISEA/build/DiffusionTest/dune-diffusionFEM/src/test");

    // calculate error
    double error = 0 ;

    // calculate standard error
    // select norm for error computation
    typedef Dune::Fem::L2Norm< GridPartType > NormType;
    NormType norm( gridPart );
    error = norm.distance( gridExactSolution, scheme.solution() );

    // write vtk file
    if (graphics) {
        Dune::Fem::SubsamplingVTKIO<GridPartType> vtkio(gridPart);

        vtkio.addVertexData(scheme.solution(), "Numerical_solution");
        vtkio.addVertexData(gridExactSolution, "Exact_solution");
        //vtkio.addVertexData(, "Dirichlet_boundary_condition");

        vtkio.write(fullname.str(), Dune::VTK::appendedraw);
    }

    return error ;
}

template<class GridType>
bool solvePoissonPDE(GridType &grid, const int refineSteps, const int level, const int repeats, const int problemNumber)
{
    // refine grid
    Dune::Fem::GlobalRefine::apply( grid, level * refineSteps );

    // setup EOC loop

    // calculate first step
    double oldError = INFINITY;

    for( int step = 1; step <= repeats; ++step )
    {
        // refine globally such that grid with is bisected
        // and all memory is adjusted correctly
        Dune::Fem::GlobalRefine::apply( grid, refineSteps );



        const double newError = algorithm( grid, step, problemNumber );
        const double eoc = log( oldError / newError ) / M_LN2;
        if( Dune::Fem::MPIManager::rank() == 0 )
        {
            std::cout << "Error step " << step << " :" << newError << std::endl;
            std::cout << "EOC( " << step <<"/"<< step-1 <<" ) = " << eoc << std::endl;
        }
        oldError = newError;
    }

}

#endif //DUNE_DIFFUSIONFEM_DIFFUSIONPDE_HH
