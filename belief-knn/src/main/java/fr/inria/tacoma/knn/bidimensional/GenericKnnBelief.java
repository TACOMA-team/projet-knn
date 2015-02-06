package fr.inria.tacoma.knn.bidimensional;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;

import java.math.BigDecimal;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GenericKnnBelief<T> implements SensorBeliefModel<T> {

    private final int k;
    private final double alpha;
    private final FrameOfDiscernment frame;
    private final List<GenericPoint<T>> points;
    private final Map<String, Double> gammaProvider;
    Function<List<MassFunction>, MassFunction> combination;
    private BiFunction<T, T, Double> distance;

    public GenericKnnBelief(List<GenericPoint<T>> points, int k, double alpha,
                            FrameOfDiscernment frame,
                            Function<List<MassFunction>, MassFunction> combination) {
        this.k = k;
        this.alpha = alpha;
        this.frame = frame;
        this.combination = combination;
        this.points = points;
        this.gammaProvider = generateGammaProvider();
    }

    private Map<String,Double> generateGammaProvider() {
        Set<String> labels = new HashSet<>();
        points.forEach(p -> labels.add(p.getLabel()));
        Map<String, Double> gammas = new HashMap<>();
        for (String label : labels) {
            List<T> pointValues = points.stream()
                    .filter(p -> p.getLabel().equals(label))
                    .map(GenericPoint::getValue).collect(Collectors.toList());
            BigDecimal average = BigDecimal.ZERO;

            int size = pointValues.size();
            for (int i = 0; i < size; i++) {
                for (int j = i; j < size; j++) {
                    average = average.add(new BigDecimal(
                            distance.apply(pointValues.get(i), pointValues.get(j))));
                }
            }
            average = average.divide(new BigDecimal(size));
            gammas.put(label, average.doubleValue());
        }
        return gammas;
    }

    private List<GenericPoint<T>> knn(T value) {
        return points.stream()
                .sorted((p1,p2) ->
                        Double.compare(distance.apply(p1.getValue(), value), distance.apply(
                                p2.getValue(), value)))
                .limit(k).collect(Collectors.toList());
    }

    private MassFunction getMassFunction(T value, GenericPoint<T> point) {
        MassFunction mass = new MassFunctionImpl(frame);
        double gamma = 1.0 / gammaProvider.get(point.getLabel());
        mass.addToFocal(frame.toStateSet(point.getLabel()),
                alpha * Math.exp(- Math.pow(distance.apply(value, point.getValue()) * gamma, 1)));
        mass.putRemainingOnIgnorance();
        return mass;
    }

    public int getK() {
        return k;
    }

    public double getAlpha() {
        return alpha;
    }

    @Override
    public MassFunction toMass(T sensorValue) {
        List<GenericPoint<T>> knn = knn(sensorValue);
        List<MassFunction> masses = knn.stream()
                .map(p -> getMassFunction(sensorValue, p))
                .collect(Collectors.toList());
        return Mass.toConsonant(combination.apply(masses));
    }


    @Override
    public MassFunction toMassWithoutValue() {
        MassFunctionImpl massFunction = new MassFunctionImpl(frame);
        massFunction.putRemainingOnIgnorance();
        return massFunction;
    }


    @Override
    public FrameOfDiscernment getFrame() {
        return frame;
    }


}
