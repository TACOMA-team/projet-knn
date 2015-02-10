package fr.inria.tacoma.knn.generic;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.LabelledPoint;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class KnnBelief<T> implements SensorBeliefModel<T> {

    private final int k;
    private final double alpha;
    private final FrameOfDiscernment frame;
    private final List<? extends LabelledPoint<T>> points;
    private final Map<String, Double> gammaProvider;
    Function<List<MassFunction>, MassFunction> combination;
    private BiFunction<T, T, Double> distance;

    public KnnBelief(List<? extends LabelledPoint<T>> points, int k, double alpha,
                     FrameOfDiscernment frame,
                     Function<List<MassFunction>, MassFunction> combination,
                     BiFunction<T, T, Double> distance) {
        this.k = k;
        this.alpha = alpha;
        this.frame = frame;
        this.combination = combination;
        this.points = points;
        this.distance = distance;
        this.gammaProvider = generateGammaProvider(distance);
    }

    private Map<String,Double> generateGammaProvider(BiFunction<T, T, Double> distance) {
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

    private List<LabelledPoint<T>> knn(T value) {
        return points.stream()
                .sorted((p1,p2) ->
                        Double.compare(distance.apply(p1.getValue(), value), distance.apply(
                                p2.getValue(), value)))
                .limit(k).collect(Collectors.toList());
    }

    private MassFunction getMassFunction(T value, LabelledPoint<T> point) {
        MassFunction mass = new MassFunctionImpl(frame);
        double gamma = 1.0 / gammaProvider.get(point.getLabel());
        mass.addToFocal(frame.toStateSet(point.getLabel()),
                alpha * Math.exp(-Math.pow(distance.apply(value, point.getValue()) * gamma, 1)));
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
        List<LabelledPoint<T>> knn = knn(sensorValue);
        List<MassFunction> masses = knn.stream()
                .map(p -> getMassFunction(sensorValue, p))
                .collect(Collectors.toList());
        return combination.apply(masses);
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
