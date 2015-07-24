package fr.inria.tacoma.knn.core;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DoubleKnnBelief implements KnnBelief<Double> {

    private final int k;
    private final double alpha;
    private final FrameOfDiscernment frame;
    private final List<LabelledPoint<Double>> sortedPoints;
    private final Map<String, Double> gammaProvider;
    private final Function<List<MassFunction>, MassFunction> combination;
    private final BiFunction<Double, Double, Double> distance;

    public DoubleKnnBelief(List<LabelledPoint<Double>> points, int k, double alpha,
                     FrameOfDiscernment frame,
                     Function<List<MassFunction>, MassFunction> combination,
                     BiFunction<Double, Double, Double> distance,
                     Map<String,Double> gammaProvider) {
        this(points, k, alpha, frame, combination, distance, gammaProvider, false);
    }

    private DoubleKnnBelief(List<LabelledPoint<Double>> points, int k, double alpha,
                           FrameOfDiscernment frame,
                           Function<List<MassFunction>, MassFunction> combination,
                           BiFunction<Double, Double, Double> distance,
                           Map<String,Double> gammaProvider, boolean sorted) {
        assert alpha > 0;
        this.k = k;
        this.alpha = alpha;
        this.frame = frame;
        this.combination = combination;
        this.distance = distance;
        this.gammaProvider = gammaProvider;
        if(sorted) {
            this.sortedPoints = points;
        }
        else {
            this.sortedPoints = new ArrayList<>(points);
            Collections.sort(sortedPoints, (p1,p2) -> Double.compare(p1.getValue(), p2.getValue()));
        }
    }

    private List<LabelledPoint<Double>> knn(Double value) {
        int startIndex = 0, endIndex = sortedPoints.size() - 1 ;
        while(endIndex - startIndex > 1) {
            int index = (startIndex + endIndex) / 2;
            if(sortedPoints.get(index).getValue() >= value) {
                endIndex = index;
            }
            else {
                startIndex = index;
            }
        }

        while (endIndex - startIndex < k) {
            if(startIndex == 0) {
                endIndex = k;
                break;
            }
            else if (endIndex == sortedPoints.size() - 1) {
                startIndex = endIndex - k;
                break;
            }

            Double startValue = sortedPoints.get(startIndex - 1).getValue();
            Double endValue = sortedPoints.get(endIndex + 1).getValue();
            if(value - startValue < endValue - value) {
                startIndex--;
            }
            else {
                endIndex++;
            }
        }

        return sortedPoints.subList(startIndex, endIndex + 1);
    }

    private MassFunction getMassFunction(Double value, LabelledPoint<Double> point) {
        MutableMass mass = frame.newMass();
        double gamma = 1.0 / gammaProvider.get(point.getLabel());
        mass.addToFocal(point.getStateSet(),
                alpha * Math.exp(- distance.apply(value, point.getValue()) * gamma));
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
    public KnnBelief<Double> withAlpha(double newAlpha) {
        return new DoubleKnnBelief(sortedPoints, k, newAlpha, frame, combination, distance,
                gammaProvider, true);
    }

    @Override
    public KnnBelief<Double> withK(int newK) {
        return new DoubleKnnBelief(sortedPoints, newK, alpha, frame, combination, distance,
                gammaProvider, true);
    }

    @Override
    public KnnBelief<Double> withAlphaAndK(double newAlpha, int newK) {
        return new DoubleKnnBelief(sortedPoints, newK, newAlpha, frame, combination, distance,
                gammaProvider, true);
    }

    @Override
    public MutableMass toMass(Double sensorValue) {
        List<LabelledPoint<Double>> knn = knn(sensorValue);
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
