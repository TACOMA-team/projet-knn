package fr.inria.tacoma.knn.util;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.LabelledPoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KnnUtils {

    /**
     * Extract the end of a list and returns the extracted list. The items will
     * be removed from the list given as an argument. The function takes a ratio
     * which is the quantity of items ( between  0.0 and 1.0) which is kept in
     * list. It is useful to split a list between the learning set and the
     * cross validation set.
     * @param list list to split
     * @param keepRatio ratio of the list to keep.
     * @return the extracted list.
     */
    public static <T> List<T> extractSubList(List<T> list, double keepRatio) {
        assert keepRatio < 1.0;
        List<T> subList = list.subList((int) ((list.size() - 1) * keepRatio), list.size() - 1);
        List<T> extracted = new ArrayList<>();
        extracted.addAll(subList);
        subList.clear();
        return extracted;
    }

    public static <T> List<List<T>> split(List<T> list, int nb) {
        int size = list.size();
        double sublistSize = (double)size / nb;

        List<List<T>> sublists = IntStream.range(0, nb)
                .mapToDouble(i -> (double) i * sublistSize) // compute intervals
                .mapToObj(start -> list.subList((int) start,
                        (int) (start + sublistSize))) // create sublist views
                .map(sublist -> new ArrayList<>(sublist)) // copy sublists to avoid troubles
                .collect(Collectors.toList());

        return sublists;
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
    public static <T> double error(List<? extends LabelledPoint<T>> crossValidation,
                                   SensorBeliefModel<T> model) {
        return crossValidation.stream().mapToDouble(point -> {
            MassFunction actualMassFunction = model.toMass(point.getValue());
            MutableMass idealMassFunction = model.getFrame().newMass();
            idealMassFunction.set(model.getFrame().toStateSet(point.getLabel()), 1);
            idealMassFunction.putRemainingOnIgnorance();
            double distance = Mass.jousselmeDistance(actualMassFunction, idealMassFunction);
            return distance * distance / crossValidation.size();
        }).sum();
    }


    public static <T> KnnBelief<T> getBestKnnBelief(FrameOfDiscernment frame,
                                                          List<? extends LabelledPoint<T>> points,
                                                          List<? extends LabelledPoint<T>> crossValidation,
                                                          double alpha,
                                                          BiFunction<T, T, Double> distance) {
        return KnnUtils.getBestKnnBelief(frame, points, crossValidation, alpha, distance,
                points.size() - 1);
    }
    public static <T> KnnBelief<T> getBestKnnBeliefForAlphaAndK(FrameOfDiscernment frame,
            List<LabelledPoint<T>> points,
            List<LabelledPoint<T>> crossValidation,
            BiFunction<T, T, Double> distance) {

        int maxNeighborCount =  points.size() - 1;

        List<KnnBelief<T>> models = getKnnBeliefsForK(frame, points, crossValidation, distance,
                maxNeighborCount);

        KnnBelief<T> bestModel = null;
        double lowestError = Double.POSITIVE_INFINITY;
        for (KnnBelief<T> model : models) {
            double error = KnnUtils.error(crossValidation, model);
            if (error < lowestError) {
                lowestError = error;
                bestModel = model;
            }
        }

        assert bestModel != null;
//        System.out.println("lowest error: " + lowestError);
//        System.out.println("bestNeighborCount: " + bestModel.getK());
//        System.out.println("best alpha: " + bestModel.getAlpha());
        return bestModel;
    }

    private static <T> List<KnnBelief<T>> getKnnBeliefsForK(FrameOfDiscernment frame,
                                                            List<? extends LabelledPoint<T>> points,
                                                            List<? extends LabelledPoint<T>> crossValidation,
                                                            BiFunction<T, T, Double> distance,
                                                            int maxNeighborCount) {
        //TODO change limit if necessary, it is just to avoid long computations
        return IntStream.range(1, maxNeighborCount).limit(50).parallel().mapToObj(
                    k -> {
                        KnnBelief<T> model = null;
                        double lowestError = Double.POSITIVE_INFINITY;
                        for (int i = 1; i < 100; i++) {
                            double alpha = 0.01 * i;
                            KnnBelief<T> beliefModel =
                                    new KnnBelief<>(points, k, alpha, frame,
                                            KnnUtils::optimizedDuboisAndPrade, distance);
                            double error = KnnUtils.error(crossValidation, beliefModel);
                            if (error < lowestError) {
                                lowestError = error;
                                model = beliefModel;
                            }
                        }
                        return model;
                    }
            ).collect(Collectors.toList());
    }

    /**
     * Finds the model having the lowest error depending on K. This iterate the knn algorithm by
     * incrementing k and calculating the error. It then return the model with the minimum error.
     *
     * @param frame            frame of discernment
     * @param points      training set to use
     * @param maxNeighborCount maximum to use for k (the effective max will be limited by the size
     *                         of the training set)
     * @return the knn belief with the lowest error depending on k
     */
    public static <T> KnnBelief<T> getBestKnnBelief(FrameOfDiscernment frame,
                                                          List<? extends LabelledPoint<T>> points,
                                                          List<? extends LabelledPoint<T>> crossValidation,
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
