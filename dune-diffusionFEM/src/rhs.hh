
#ifndef DUNE_DIFFUSIONFEM_RHS_HH
#define DUNE_DIFFUSIONFEM_RHS_HH


#include <dune/fem/quadrature/cachingquadrature.hh>


// assembleRHS
// -----------

template< class Function, class DiscreteFunction >
void assembleRHS ( const Function &function, DiscreteFunction &rhs )
{
    typedef typename DiscreteFunction::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
    typedef typename DiscreteFunction::LocalFunctionType LocalFunctionType;

    typedef typename DiscreteFunctionSpaceType::IteratorType IteratorType;
    typedef typename IteratorType::Entity EntityType;
    typedef typename EntityType::Geometry GeometryType;

    typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
    typedef Dune::Fem::CachingQuadrature< GridPartType, 0 > QuadratureType;

    rhs.clear();

    const DiscreteFunctionSpaceType &dfSpace = rhs.space();

    const IteratorType end = dfSpace.end();
    for( IteratorType it = dfSpace.begin(); it != end; ++it )
    {
        const EntityType &entity = *it;
        const GeometryType &geometry = entity.geometry();

        typename Function::LocalFunctionType localFunction =
                function.localFunction( entity);
        LocalFunctionType rhsLocal = rhs.localFunction( entity );

        QuadratureType quadrature( entity, 2*dfSpace.order()+1 );
        const size_t numQuadraturePoints = quadrature.nop();
        for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
        {
            // obtain quadrature point
            const typename QuadratureType::CoordinateType &x = quadrature.point( pt );

            // evaluate f
            typename Function::RangeType f;
            localFunction.evaluate( quadrature[ pt ], f );

            // multiply by quadrature weight
            f *= quadrature.weight( pt ) * geometry.integrationElement( x );

            // add f * phi_i to rhsLocal[ i ]
            rhsLocal.axpy( quadrature[ pt ], f );
        }
    }
    rhs.communicate();
}

// assembleRHS
// -----------

template< class Model, class DiscreteFunction >
void assembleDGRHS ( const Model &model, DiscreteFunction &rhs )
{
    assembleRHS( model.rightHandSide(), rhs );
    if ( ! model.hasDirichletBoundary() )
        return;

    typedef typename DiscreteFunction::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
    typedef typename DiscreteFunction::LocalFunctionType LocalFunctionType;

    typedef typename DiscreteFunctionSpaceType::IteratorType IteratorType;
    typedef typename IteratorType::Entity EntityType;
    typedef typename EntityType::Geometry GeometryType;
    typedef typename LocalFunctionType::RangeType RangeType;
    typedef typename LocalFunctionType::JacobianRangeType JacobianRangeType;
    typedef typename DiscreteFunctionSpaceType::DomainType DomainType;
    static const int dimDomain = LocalFunctionType::dimDomain;
    static const int dimRange = LocalFunctionType::dimRange;

    typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
    typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
    typedef typename IntersectionIteratorType::Intersection IntersectionType;

    typedef Dune::Fem::ElementQuadrature< GridPartType, 1 > FaceQuadratureType;

    const DiscreteFunctionSpaceType &dfSpace = rhs.space();
    const int quadOrder = 2*dfSpace.order()+1;

    const IteratorType end = dfSpace.end();
    for( IteratorType it = dfSpace.begin(); it != end; ++it )
    {
        const EntityType &entity = *it;
        if ( !entity.hasBoundaryIntersections() )
            continue;

        const GeometryType &geometry = entity.geometry();
        double area = geometry.volume();

        LocalFunctionType rhsLocal = rhs.localFunction( entity );

        const IntersectionIteratorType iitend = dfSpace.gridPart().iend( entity );
        for( IntersectionIteratorType iit = dfSpace.gridPart().ibegin( entity ); iit != iitend; ++iit ) // looping over intersections
        {
            const IntersectionType &intersection = *iit;
            if ( ! intersection.boundary() ) // i.e. if intersection is on boundary: nothing to be done for Neumann zero b.c.
                continue;                      // since [u] = 0  and grad u.n = 0
            if ( ! model.isDirichletIntersection( intersection ) )
                continue;

            typedef typename IntersectionType::Geometry  IntersectionGeometryType;
            const IntersectionGeometryType &intersectionGeometry = intersection.geometry();

            const double intersectionArea = intersectionGeometry.volume();
            const double beta = model.penalty() * intersectionArea / area;

            FaceQuadratureType quadInside( dfSpace.gridPart(), intersection, quadOrder, FaceQuadratureType::INSIDE );
            const size_t numQuadraturePoints = quadInside.nop();
            for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
            {
                const typename FaceQuadratureType::LocalCoordinateType &x = quadInside.localPoint( pt );
                const DomainType normal = intersection.integrationOuterNormal( x );

                const double weight = quadInside.weight( pt );

                RangeType value;
                JacobianRangeType dvalue,advalue;

                RangeType vuOut;
                model.g( RangeType(0), entity, quadInside.point(pt), vuOut );

                value = vuOut;
                value *= beta * intersectionGeometry.integrationElement( x );

                //  [ u ] * { grad phi_en } = -normal(u+ - u-) * 0.5 grad phi_en
                // diadic product of u x n
                for (int r=0;r<dimRange;++r)
                    for (int d=0;d<dimDomain;++d)
                        dvalue[r][d] = -0.5 * normal[d] * vuOut[r];

                model.diffusiveFlux( entity, quadInside[ pt ], vuOut, dvalue, advalue );

                value *= weight;
                advalue *= weight;
                rhsLocal.axpy( quadInside[ pt ], value, advalue );
            }
        }
    }
    rhs.communicate();
}

#endif //DUNE_DIFFUSIONFEM_RHS_HH
