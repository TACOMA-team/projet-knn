package fr.inria.tacoma.bft.combinations;

import fr.inria.tacoma.bft.combinations.internal.CombinationIterator;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public final  class Combinations {

    private Combinations(){}

    /**
     * Dempster's rule of combination.
     * With m1, m2 the two mass function we want to combine and m12 the resulting
     * mass function, this rule is defined as :
     * <br/>
     * m12(∅) = 0
     * <br/>
     * m12(X) = (∑ {m1(A)*m2(B) | A ∪ B = X ≠ ∅ }) * 1 / (1- K)
     * <br/>
     * with K = ∑ {m1(A)*m2(B) | A ∪ B = ∅ }
     *
     * @param mass1 first mass to combine
     * @param mass2 second mass to combine
     * @return combined mass
     */

    public static MassFunction dempster(MassFunction mass1, MassFunction mass2) {

        FrameOfDiscernment frame = mass1.getFrameOfDiscernment();
        MassFunction result = new MassFunctionImpl(frame);

        mass1.foreachFocalElement((setId1, value1) -> {
            mass2.foreachFocalElement((setId2, value2) -> {
                StateSet stateSet = setId1.conjunction(setId2);
                if (!stateSet.isEmpty()) {
                    double value = value1 * value2;
                    result.addToFocal(stateSet, value);
                }
            });
        });

        if (result.getTotalAssignedMass() == 0) {
            throw new IllegalArgumentException("Mass functions " + mass1 + " and " +
                    mass2 + " are in full conflict. The dempster combination is not defined for full" +
                    "conflicting functions.");
        }
        result.normalize();
        return result;
    }

    /**
     * Dubois and Prade's rule of combination.
     * With m1, m2 the two mass function we want to combine and m12 the resulting
     * mass function, this rule is defined as :
     * <br />
     * m12(X) = ∑ {m1(A)*m2(B) | A ∩ B = X } + ∑ {m1(A)*m2(B) | A ∪ B = X and A ∩ B = ∅ }
     *
     * @param mass1 first mass to combine
     * @param mass2 second mass to combine
     * @return combined mass
     */
    public static MassFunction duboisAndPrade(MassFunction mass1, MassFunction mass2) {

        FrameOfDiscernment frame = mass1.getFrameOfDiscernment();
        MassFunction combination = new MassFunctionImpl(frame);

        mass1.foreachFocalElement((set1, value1) -> {
            mass2.foreachFocalElement((set2, value2) -> {
                StateSet conjunction = set1.conjunction(set2);
                if(!conjunction.isEmpty()) {
                    combination.addToFocal(conjunction, value1 * value2);
                }
                else {
                    StateSet disjunction = set1.disjunction(set2);
                    combination.addToFocal(disjunction, value1 * value2);
                }
            });
        });

        return combination;
    }

    /**
     * Dubois and Prade's rule of combination for more than 2 mass functions.
     * This combination rule with 2 mass function is not commutative (unlike
     * Dempster's rule of combination). The only way to do it is considering
     * a set of function as a whole and combine them together. The order of
     * the list does not matter.
     *
     * @param massFunctions list of mass function to combine
     * @return the combined mass function
     */
    public static MassFunction duboisAndPrade(List<MassFunction> massFunctions) {
        FrameOfDiscernment frame = massFunctions.get(0).getFrameOfDiscernment();
        MassFunction combination = new MassFunctionImpl(frame);

        List<Collection<StateSet>> StateSetsCollections = massFunctions.stream()
                .map(MassFunction::getFocalStateSets)
                .collect(Collectors.toList());

        CombinationIterator<StateSet> iterator = new CombinationIterator<>(StateSetsCollections);
        while (iterator.hasNext()) {
            List<StateSet> stateSets = iterator.next();

            StateSet conjunction = stateSets.stream().reduce((a,b) -> a.conjunction(b)).get();
            double mass = 1;
            for (int i = 0; i < massFunctions.size(); i++) {
                mass *= massFunctions.get(i).get(stateSets.get(i));
            }

            if(!conjunction.isEmpty()) {
                combination.addToFocal(conjunction, mass);
            }
            else {
                StateSet disjunction = stateSets.stream().reduce((a,b)->a.disjunction(b)).get();
                combination.addToFocal(disjunction, mass);
            }
        }

        return combination;
    }


}
