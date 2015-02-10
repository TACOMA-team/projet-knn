package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.mass.MassFunction;

import java.util.ArrayList;
import java.util.List;

public class KnnUtils {

    /**
     * Extract the end of a list and returns the extracted list. The items will
     * be removed from the list given as an argument. The function takes a ratio
     * which is the quantity of items ( between  0.0 and 1.0) which is kept in
     * list. It is useful to split a list between the learning set and the
     * cross validation set.
     * @param list list to split
     * @param ratio ratio of the list to keep.
     * @return the extracted list.
     */
    public static <T> List<T> extractSubList(List<T> list, double ratio) {
        assert ratio < 1.0;
        List<T> subList = list.subList((int)((list.size() - 1) * ratio), list.size() - 1);
        List<T> extracted = new ArrayList<>();
        extracted.addAll(subList);
        subList.clear();
        return extracted;
    }


    /**
     * An hybrid fusion mecanism which apply dempster for every points with the same label, end the
     * fuse the resulting mass functions with dubois and prade. This allow to perform a very
     * efficient dubois and prade.
     *
     * @param masses masses to fuse
     * @return fused mass function
     */
    public static MassFunction optimizedDuboisAndPrade(List<MassFunction> masses) {
        List<MassFunction> optimizedMasses = new ArrayList<>(masses);
        for (int refMassIndex = 0; refMassIndex < optimizedMasses.size(); refMassIndex++) {
            MassFunction referenceMass = optimizedMasses.get(refMassIndex);
            for (int j = refMassIndex + 1; j < optimizedMasses.size(); ) {
                MassFunction mass2 = optimizedMasses.get(j);
                if (referenceMass.getFocalStateSets().equals(mass2.getFocalStateSets())) {
                    referenceMass = Combinations.dempster(referenceMass, mass2);
                    optimizedMasses.remove(j);
                } else {
                    j++;
                }
            }
            optimizedMasses.set(refMassIndex, referenceMass);
        }
        return Combinations.duboisAndPrade(optimizedMasses);
    }

}
