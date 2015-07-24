package fr.inria.tacoma.knn.core;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GenericKnn<T> implements KnnBelief<T> {

    private final int k;
    private final double alpha;
    private final FrameOfDiscernment frame;
    private final List<? extends LabelledPoint<T>> points;
    private final Map<String, Double> gammaProvider;
    private final Function<List<MassFunction>, MassFunction>combination;
    private final BiFunction<T, T, Double> distance;

    public GenericKnn(List<? extends LabelledPoint<T>> points, int k, double alpha,
                      FrameOfDiscernment frame,
                      Function<List<MassFunction>, MassFunction> combination,
                      BiFunction<T, T, Double> distance,
                      Map<String, Double> gammaProvider) {
        this.k = k;
        this.alpha = alpha;
        this.frame = frame;
        this.combination = combination;
        this.points = points;
        this.distance = distance;
        this.gammaProvider = gammaProvider;
    }

    private List<LabelledPoint<T>> knn(T value) {
        return points.stream()
                .sorted((p1,p2) ->
                        Double.compare(distance.apply(p1.getValue(), value), distance.apply(
                                p2.getValue(), value)))
                .limit(k).collect(Collectors.<LabelledPoint<T>>toList());
    }

    private MassFunction getMassFunction(T value, LabelledPoint<T> point) {
        MutableMass mass = frame.newMass();
        double gamma = 1.0 / gammaProvider.get(point.getLabel());
        mass.addToFocal(frame.toStateSet(point.getLabel()),
                alpha * Math.exp(-distance.apply(value, point.getValue()) * gamma));
        mass.putRemainingOnIgnorance();
        return mass;
    }

    @Override
    public int getK() {
        return k;
    }

    @Override
    public double getAlpha() {
        return alpha;
    }

    @Override
    public Map<String, Double> getGammas() {
        return gammaProvider;
    }

    @Override
    public KnnBelief<T> withAlpha(double newAlpha) {
        return new GenericKnn<>(points, k, newAlpha, frame, combination, distance, gammaProvider);
    }

    @Override
    public KnnBelief<T> withK(int newK) {
        return new GenericKnn<>(points, newK, alpha, frame, combination, distance, gammaProvider);
    }

    @Override
    public KnnBelief<T> withAlphaAndK(double newAlpha, int newK) {
        return new GenericKnn<>(points, newK, newAlpha, frame, combination, distance, gammaProvider);
    }

    @Override
    public MutableMass toMass(T sensorValue) {
        List<LabelledPoint<T>> knn = knn(sensorValue);
        List<MassFunction> masses = knn.stream()
                .map(p -> getMassFunction(sensorValue, p))
                .collect(Collectors.toList());
        return frame.newMass(combination.apply(masses));
    }


    @Override
    public MassFunction toMassWithoutValue() {
        MutableMass massFunction = frame.newMass();
        massFunction.putRemainingOnIgnorance();
        return massFunction;
    }


    @Override
    public FrameOfDiscernment getFrame() {
        return frame;
    }


}
