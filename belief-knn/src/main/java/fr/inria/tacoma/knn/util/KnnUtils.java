package fr.inria.tacoma.knn.util;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.criteria.Criteria;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.unidimensional.SensorValue;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KnnUtils {

    public static final double NEWTON_STEP = 0.0001;
    public static final double MAX_ALPHA = 0.9;

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
     * An hybrid fusion mechanism which apply dempster for every points with the same label, end the
     * fuse the resulting mass functions with dubois and prade. This allow to perform a very
     * efficient dubois and prade.
     *
     * @param masses masses to fuse
     * @return fused mass function
     */
    public static MassFunction optimizedDuboisAndPrade(List<MassFunction> masses) {
        Map<Set<StateSet>, MassFunction> optimized = new HashMap<>();

        for (MassFunction mass : masses) {
            optimized.compute(mass.getFocalStateSets(), (k,v) -> {
                if(v == null) {
                    return mass;
                }
                else {
                    return Combinations.dempster(mass, v);
                }
            });
        }
        return Combinations.duboisAndPrade(new ArrayList<>(optimized.values()));
    }

    /**
     * An hybrid fusion mechanism which apply dempster for every points with the same label, end the
     * fuse the resulting mass functions with dubois and prade. This allow to perform a very
     * efficient dubois and prade.
     *
     * @param masses masses to fuse
     * @return fused mass function
     */
    public static MassFunction optimizedDuboisAndPrade2(List<MassFunction> masses) {
        Map<Set<StateSet>, MassFunction> optimized = new HashMap<>();

        for (MassFunction mass : masses) {
            optimized.compute(mass.getFocalStateSets(), (k,v) -> {
                if(v == null) {
                    return mass;
                }
                else {
                    return Combinations.dempster(mass, v);
                }
            });
        }

        System.out.println(masses);
        return Combinations.duboisAndPrade(new ArrayList<>(optimized.values()));
    }

    public static MassFunction optimizedDempster(List<MassFunction> masses) {
        Map<Set<StateSet>, MassFunction> optimized = new HashMap<>();

        for (MassFunction mass : masses) {
            optimized.compute(mass.getFocalStateSets(), (k,v) -> {
                if(v == null) {
                    return mass;
                }
                else {
                    return Combinations.dempster(mass, v);
                }
            });
        }

        return optimized.values().stream().reduce(Combinations::dempster).get();
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
            MutableMass idealMassFunction = model.getFrame().newMass()
                    .set(point.getStateSet(), 1.0)
                    .putRemainingOnIgnorance();
            double distance = Mass.jousselmeDistance(actualMassFunction, idealMassFunction);
            return distance * distance;
        }).average().orElse(0);
    }


    public static <T> KnnBelief<T> getBestKnnBeliefWithFixedAlpha(KnnFactory<T> factory,
                                                                  List<? extends LabelledPoint<T>> points,
                                                                  List<? extends LabelledPoint<T>> crossValidation,
                                                                  double alpha) {
        return KnnUtils.getBestKnnBeliefWithFixedAlpha(factory, points, crossValidation, alpha,
                points.size() - 1);
    }

    public static <T> KnnBelief<T> getBestKnnBeliefForAlphaAndK(KnnFactory<T> factory,
            List<LabelledPoint<T>> points,
            List<LabelledPoint<T>> crossValidation) {

        int maxNeighborCount =  points.size() - 1;
        List<KnnBelief<T>> models = getKnnBeliefsForK(factory, points, crossValidation,
                maxNeighborCount);

        KnnBelief<T> bestModel = null;
        double lowestError = Double.POSITIVE_INFINITY;
        for (KnnBelief<T> model : models) {
            double error = KnnUtils.error(crossValidation, model);
//            System.out.println(error);
            if (error < lowestError) {
                lowestError = error;
                bestModel = model;
            }
        }

        assert bestModel != null;
        System.out.println("lowest error: " + lowestError);
        System.out.println("bestNeighborCount: " + bestModel.getK());
        System.out.println("best alpha: " + bestModel.getAlpha());
        return bestModel;
    }

    private static <T> List<KnnBelief<T>> getKnnBeliefsForK(KnnFactory<T> factory,
                                                            List<? extends LabelledPoint<T>> points,
                                                            List<? extends LabelledPoint<T>> crossValidation,
                                                            int maxNeighborCount) {
        return IntStream.range(2, maxNeighborCount).limit(100).mapToObj(
                k -> getBestModelForFixedKNewton(factory, points, crossValidation, k)
        ).collect(Collectors.toList());
    }

    private static <T> KnnBelief<T> getBestModelForFixedK(KnnFactory<T> factory,
                                                          List<? extends LabelledPoint<T>> points,
                                                          List<? extends LabelledPoint<T>> crossValidation,
                                                          int k) {
        Map<String, Double> gammas = generateGammaProvider(factory.getDistance(), points);
        KnnBelief<T> model = null;
        double lowestError = Double.POSITIVE_INFINITY;
        for (int i = 1; i < 100; i++) {
            double alpha = 0.01 * i;
            KnnBelief<T> beliefModel = factory.newKnnBelief(points, gammas, k, alpha);
            double error = KnnUtils.error(crossValidation, beliefModel);
//            System.out.println(alpha+";"+error);
            if (error < lowestError) {
                lowestError = error;
                model = beliefModel;
            }
        }
        return model;
    }

    public static <T> KnnBelief<T> getBestModelForFixedKNewton(KnnFactory<T> factory,
                                                          List<? extends LabelledPoint<T>> points,
                                                          List<? extends LabelledPoint<T>> crossValidation,
                                                          int k) {
        Map<String, Double> gammas = generateGammaProvider(factory.getDistance(), points);
        KnnBelief<T> model = null;
        double alpha = 0.05;
        int iterations = 0;
        double stopCriteria = 0.001;
        double variation = stopCriteria + 2;
//        System.out.println("k=" + k);
        while(iterations < 10 && variation > stopCriteria) {
            model = factory.newKnnBelief(points, gammas, k, alpha);
            variation = computeVariation(crossValidation, model, alpha);
//            System.out.println(alpha + " - " + variation + " -> " +  (alpha - variation));
            alpha = alpha - variation;
            if(alpha < NEWTON_STEP) {
                alpha = 2.0 * NEWTON_STEP;
            }
            else if(alpha > MAX_ALPHA) {
                alpha = MAX_ALPHA;
            }
            iterations++;
        }
//        System.out.println();
        return factory.newKnnBelief(points, gammas, k, alpha);
    }

    private static <T> double computeVariation(List<? extends LabelledPoint<T>> crossValidation,
                                               KnnBelief<T> model, double alpha) {

//        try {

            double errorLeft = KnnUtils.error(crossValidation, model.withAlpha(alpha - NEWTON_STEP));
            double errorRight = KnnUtils.error(crossValidation, model.withAlpha(alpha + NEWTON_STEP));
            double errorCenter = KnnUtils.error(crossValidation, model.withAlpha(alpha));

            double diffCenter = (errorRight - errorLeft) / (2 * NEWTON_STEP);
            double secondOrder = (errorRight + errorLeft - 2 * errorCenter) / (NEWTON_STEP * NEWTON_STEP);
            return diffCenter / secondOrder;
//        }
//        catch (Exception e) {
//            printErrorForModel(crossValidation, model);
//            throw e;
//        }
    }

    public static <T> void printErrorForModel(List<? extends LabelledPoint<T>> crossValidation,
                                               KnnBelief<T> model, PrintWriter writer) {
        for (int i = 1; i < 100; i++) {
            double alpha1 = 0.01 * i;
            KnnBelief<T> beliefModel = model.withAlpha(alpha1);
            double error = KnnUtils.error(crossValidation, beliefModel);
            writer.println(alpha1+";"+error);
        }
    }

    /**
     * Finds the model having the lowest error depending on K. This iterate the knn algorithm by
     * incrementing k and calculating the error. It then return the model with the minimum error.
     *
     * @param points      training set to use
     * @param maxNeighborCount maximum to use for k (the effective max will be limited by the size
     *                         of the training set)
     * @return the knn belief with the lowest error depending on k
     */
    public static <T> KnnBelief<T> getBestKnnBeliefWithFixedAlpha(KnnFactory<T> factory,
                                                                  List<? extends LabelledPoint<T>> points,
                                                                  List<? extends LabelledPoint<T>> crossValidation,
                                                                  double alpha,
                                                                  int maxNeighborCount) {
        double lowestError = Double.POSITIVE_INFINITY;
        KnnBelief<T> bestModel = null;
        Map<String, Double> gammas = generateGammaProvider(factory.getDistance(), points);

        maxNeighborCount = Math.min(maxNeighborCount, points.size() - 1);
        for (int neighborCount = 1; neighborCount <= maxNeighborCount; neighborCount++) {
            KnnBelief<T> beliefModel = factory.newKnnBelief(points, gammas, neighborCount, alpha);
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


    public static <T> Map<String,Double> generateGammaProvider(BiFunction<T, T, Double> distance,
                                                         List<? extends LabelledPoint<T>> points) {
        Set<String> labels = new HashSet<>();
        points.forEach(p -> labels.add(p.getLabel()));
        Map<String, Double> gammas = new HashMap<>();
        for (String label : labels) {
            List<T> pointValues = points.stream()
                    .filter(p -> p.getLabel().equals(label))
                    .map(LabelledPoint::getValue).collect(Collectors.toList());
            BigDecimal average = BigDecimal.ZERO;

            int size = pointValues.size();
            for (int i = 0; i < size; i++) {
                for (int j = i + 1; j < size; j++) {
                    average = average.add(new BigDecimal(
                                    distance.apply(pointValues.get(i), pointValues.get(j))
                            )
                    );
                }
            }
            average = average.divide(new BigDecimal(size * (size - 1)), new MathContext(10));
            gammas.put(label, average.doubleValue());
        }
        return gammas;
    }

    public static List<LabelledPoint<Double>> parseData(FrameOfDiscernment frame,
                                                        String... files) throws IOException {
        List<LabelledPoint<Double>> data = new ArrayList<>();
        for (String file : files) {
            int fileOffSet = file.lastIndexOf('/');
            String label = file.substring(fileOffSet + 1, file.indexOf('-', fileOffSet));
            data.addAll(getPoints(label, file, frame));
        }
        return data;
    }

    public static List<LabelledPoint<Double>> parseData(FrameOfDiscernment frame,
                                                        Function<List<LabelledPoint<Double>>, List<LabelledPoint<Double>>> transformation,
                                                        String... files) throws IOException {
        List<LabelledPoint<Double>> data = new ArrayList<>();
        for (String file : files) {
            int fileOffSet = file.lastIndexOf('/');
            String label = file.substring(fileOffSet + 1, file.indexOf('-', fileOffSet));
            data.addAll(getPoints(label, file, frame, transformation));
        }
        return data;
    }

    /**
     * Parse the given file to extract points. One file is expected to contain only one sensor.
     *
     * @param label state in which the sample was take (i.e. presence or absence)
     * @param file  path to the file
     * @return list of points in the file
     * @throws IOException
     */
    public static List<LabelledPoint<Double>> getPoints(String label, String file,
                                                        FrameOfDiscernment frame) throws IOException {
        return getPoints(label, file, frame, list -> list);
    }

    private static List<LabelledPoint<Double>> getPoints(String label, String file,
                                                         FrameOfDiscernment frame,
                                                         Function<List<LabelledPoint<Double>>,
                                                                 List<LabelledPoint<Double>>> transformation)
            throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, true);
        InputStream resourceAsStream = Thread.currentThread().getContextClassLoader()
                .getResourceAsStream(file);
        MappingIterator<SensorValue> iterator = mapper.readValues(
                new JsonFactory().createParser(resourceAsStream),
                SensorValue.class);
        List<LabelledPoint<Double>> points = new ArrayList<>();
        while (iterator.hasNext()) {
            SensorValue next = iterator.next();
            next.setLabel(label);
            next.setStateSet(frame.toStateSet(next.getLabel()));
            points.add(next);
        }
        return transformation.apply(points);
    }
}
