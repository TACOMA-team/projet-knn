package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;
import fr.inria.tacoma.knn.generic.KnnBelief;
import fr.inria.tacoma.knn.generic.Point;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

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

    /**
     * Computes an error according to a cross validation set of points and a
     * given model. The given error is the sum of the squared distance between
     * the ideal mass function the model should have returned, and the actual
     * function. The used distance is the Jousselme distance. The "ideal" mass
     * function is a function with all the mass assigned to the label of the
     * point.
     * @param crossValidation list of points to use
     * @param model model to check
     * @param <T> type of data used by the model
     * @return the error
     */
    public static <T> double error(List<? extends Point<T>> crossValidation,
                                   SensorBeliefModel<T> model) {
        return crossValidation.stream().mapToDouble(point -> {
            MassFunction actualMassFunction = model.toMass(point.getValue());
            MassFunction idealMassFunction = new MassFunctionImpl(model.getFrame());
            idealMassFunction.set(model.getFrame().toStateSet(point.getLabel()), 1);
            idealMassFunction.putRemainingOnIgnorance();
            double distance = Mass.jousselmeDistance(actualMassFunction, idealMassFunction);
            return distance * distance;
        }).sum();
    }


    public static <T> KnnBelief<T> getBestKnnBelief(FrameOfDiscernment frame,
                                                          List<? extends Point<T>> points,
                                                          List<? extends Point<T>> crossValidation,
                                                          double alpha,
                                                          BiFunction<T, T, Double> distance) {
        return KnnUtils.getBestKnnBelief(frame, points, crossValidation, alpha, distance,
                points.size() - 1);
    }

    /**
     * Finds the model having the lowest error depending on K. This iterate the knn algorithm by
     * incrementing k and calculating the error. It then return the model with the minimum error.
     *
     * @param frame            frame of discernment
     * @param points      training set to use
     * @param maxNeighborCount maximum to use for k (the effective max will be limited by the size
     *                         of the training set)
     * @return
     */
    public static <T> KnnBelief<T> getBestKnnBelief(FrameOfDiscernment frame,
                                                          List<? extends Point<T>> points,
                                                          List<? extends Point<T>> crossValidation,
                                                          double alpha,
                                                          BiFunction<T, T, Double> distance,
                                                          int maxNeighborCount) {
        double lowestError = Double.POSITIVE_INFINITY;
        KnnBelief<T> bestModel = null;

        maxNeighborCount = Math.min(maxNeighborCount, points.size() - 1);
        for (int neighborCount = 1; neighborCount <= maxNeighborCount; neighborCount++) {
            KnnBelief<T> beliefModel =
                    new KnnBelief<>(points, neighborCount, alpha, frame,
                            KnnUtils::optimizedDuboisAndPrade, distance);
            double error = KnnUtils.error(crossValidation, beliefModel);
            if (error < lowestError) {
                lowestError = error;
                bestModel = beliefModel;
            }
        }
        assert bestModel != null;
        System.out.println("lowest error: " + lowestError);
        System.out.println("bestNeighborCount: " + bestModel.getK());
        return bestModel;
    }

}
