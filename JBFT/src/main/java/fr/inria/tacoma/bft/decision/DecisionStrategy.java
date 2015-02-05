package fr.inria.tacoma.bft.decision;

import fr.inria.tacoma.bft.core.mass.MassFunction;


/**
 * This interface is used to choose a state from a mass function by
 * extracting a State set wich seems the most probable given a mass function.
 */
public interface DecisionStrategy {

    /**
     * Choose a state set and a confidence corresponding to the given mass
     * function.
     * @param massFunction mass function to work with
     * @return a Decision object containing the chosen state set and confidence.
     */
    Decision decide(MassFunction massFunction);
}
