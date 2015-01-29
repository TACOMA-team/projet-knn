package fr.inria.tacoma.bft.combinations;

import fr.inria.tacoma.bft.core.mass.MassFunction;

import java.util.function.BinaryOperator;

/**
 * This interface represents a belief combination. It is simply a binary
 * operator as defined by the standard java interface BinaryOperator
 * which applies on Mass functions.
 */
public interface BeliefCombination extends BinaryOperator<MassFunction> {

}
