package fr.inria.tacoma.bft.criteria;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;

/**
 * Utility class. This class contains the common criteria used in the
 * belief function theory.
 */
public final class Criteria {

    private Criteria() {}

    /**
     * Computes the plausibility criterion for the given set.
     * Plausibility is defined as:
     * pl(A) = ∑ {m(B) | B ∩ A ≠ ∅ }
     * @param massFunction mass function for which we want to apply the criterion
     * @param stateSet set for which we want to compute the criterion
     * @return plausibility criterion as defined in the belief function theory
     */
    public static double plausibility(MassFunction massFunction, StateSet stateSet) {
        return massFunction.getFocalStateSets().stream()
                .filter(set -> !stateSet.conjunction(set).isEmpty())
                .mapToDouble(massFunction::get).sum();
    }

    /**
     * Computes the bet on probability (or pignistic transformation) criterion
     * for the given set.
     * BetP is defined as:
     * BetP(A) = ∑ {|A∩B| / |B| * m(B) | B ≠ ∅ }
     * @param massFunction mass function for which we want to apply the criterion
     * @param stateSet set for which we want to compute the criterion
     * @return BetP criterion as defined in the belief function theory
     */
    public static double betP(MassFunction massFunction, StateSet stateSet) {
        return massFunction.getFocalStateSets().stream()
                .filter(set -> !set.isEmpty())
                .mapToDouble(set -> {
                    double intersectionCard = (double) stateSet.conjunction(set).card();
                    return (intersectionCard / set.card() * massFunction.get(set));
                }).sum();
    }

    /**
     * Computes the Belief criterion for the given set according
     * to the mass.
     * Belief is defined as:
     * bel(A) = ∑ {m(B) | B ≠ ∅ and B ⊆ A}
     * @param massFunction mass function for which we want to apply the criterion
     * @param stateSet set for which we want to compute the criterion
     * @return belief criterion as defined in the belief function theory
     */
    public static double belief(MassFunction massFunction, StateSet stateSet) {
        return massFunction.getFocalStateSets().stream()
                .filter(set ->!set.isEmpty() && stateSet.includesOrEquals(set))
                .mapToDouble(massFunction::get).sum();
    }
}
