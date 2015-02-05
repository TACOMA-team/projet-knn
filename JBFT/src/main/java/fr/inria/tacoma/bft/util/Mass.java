package fr.inria.tacoma.bft.util;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.criteria.Criteria;

import java.util.*;

/**
 * This class contains some utility function which operate on mass functions.
 */
public final class Mass {

    private Mass(){}

    /**
     * Computes the specificity of a mass function. THe specificity shows the
     * precision of a mass function<br/>
     * The specificity is defined as Sm = ∑ { m(A) / |A| | A ≠ ∅ }.
     * @param massFunction mass on which we want to compute the specificity
     * @return specificity as defined in the theory.
     */
    public static double specificity(MassFunction massFunction) {
        return massFunction.getFocalStateSets().stream()
                .mapToDouble(set -> massFunction.get(set) / set.card())
                .sum();
    }

    /**
     * A scalar product as defined by Jousselme et al.
     * @param mass1 first mass
     * @param mass2 second mass
     * @return scalar product of the two masses
     */
    public static double scalarProduct(MassFunction mass1, MassFunction mass2) {
        double result = 0;
        for (StateSet stateSet1 : mass1.getFocalStateSets()) {
            for (StateSet stateSet2 : mass2.getFocalStateSets()) {
                double massProduct = mass1.get(stateSet1) * mass2.get(stateSet2);
                double jaccardIndex = (double) stateSet1.conjunction(stateSet2).card()
                        / (double) stateSet1.disjunction(stateSet2).card();
                result += massProduct * jaccardIndex;
            }
        }
        return result;
    }

    /**
     * Distance between two mass functions as defined by Jousselme et al.
     * @param mass1 first mass
     * @param mass2 second mass
     * @return distance between the two mass functions.
     */
    public static double jousselmeDistance(MassFunction mass1, MassFunction mass2) {
        return Math.sqrt((scalarProduct(mass1,mass1) + scalarProduct(mass2, mass2)
                - 2.0 * scalarProduct(mass1, mass2)) / 2) ;
    }

    /**
     * Transform a non consonant mass function (probabilistic mass function for
     * for instance) to a consonant mass function. This function return a new
     * mass function and does not affect the given mass function. The resulting
     * mass function will have the same pignistic transformation (betP) as the
     * original. If the given function is already consonant, it will be returned
     * as is (with no copy).
     * @param massFunction a mass function te be transformed
     * @return a consonant mass function with the same betP
     */
    public static MassFunction toConsonant(MassFunction massFunction) {
        if (isConsonant(massFunction)) { //no need to do anything
            return massFunction;
        }
        FrameOfDiscernment frame = massFunction.getFrameOfDiscernment();
        //This is not really efficient.
        TreeMap<StateSet, Double> map =
                new TreeMap<>((set1, set2) ->
                                      Double.compare(Criteria.betP(massFunction, set1),
                                                     Criteria.betP(massFunction, set2)));
        MassFunction consonantMass = new MassFunctionImpl(frame);
        for (StateSet stateSet : frame.getStateSetsWithCard(1)) {
            map.put(stateSet, Criteria.betP(massFunction, stateSet));
        }

        StateSet currentSet = frame.fullIgnoranceSet();
        double removedMass = 0;
        int setNb = map.size();
        for (Map.Entry<StateSet, Double> entry : map.entrySet()) {
            Double mass = entry.getValue() - removedMass;
            consonantMass.set(currentSet, mass * setNb);
            removedMass+= mass;
            setNb--;
            currentSet = currentSet.difference(entry.getKey());
        }

        return consonantMass;
    }

    /**
     * Checks if a mass function is consonant. A consonant mass function
     * contains only focal elements which are nested.
     * @param massFunction
     * @return
     */
    public static boolean isConsonant(MassFunction massFunction) {
        List<StateSet> list = new ArrayList<>(massFunction.getFocalStateSets());
        Collections.sort(list);
        for (int i = 1; i < list.size(); i++) {
            if(!list.get(i).includesOrEquals(list.get(i - 1))) {
                return false;
            }
        }
        return true;
    }

}
