package fr.inria.tacoma.bft.criteria;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;

/**
 * Interface for criteria. A criterion associate a state set
 * to a number depending on a mass function.
 */
@FunctionalInterface
public interface Criterion {
    double apply(MassFunction massFunction, StateSet stateSet);
}
